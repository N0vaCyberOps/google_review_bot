import os
import random
import smtplib
import sqlite3
import logging
import pickle
import time
import threading
import schedule
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime, timedelta
from typing import TypedDict, Literal

from dotenv import load_dotenv
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions  # Do łapania błędów API

# --- Konfiguracja ---
load_dotenv() # Ładuje zmienne z pliku .env

# Zapewnienie spójnych wyników dla detekcji języka
DetectorFactory.seed = 0

# Konfiguracja modelu sentymentu
MODEL_NAME = "sdadas/polish-roberta-large-sentiment"
sentiment_pipeline = None # Zostanie zainicjalizowany przy pierwszym użyciu

# Google API Scopes (wymagane uprawnienia)
SCOPES = ['https://www.googleapis.com/auth/business.manage']
API_SERVICE_NAME = 'mybusinessbusinessinformation' # Dla pobierania informacji
API_VERSION_INFO = 'v1'
API_SERVICE_REVIEWS = 'mybusinessreviews' # Dla pobierania i odpowiadania na recenzje
API_VERSION_REVIEWS = 'v1'
TOKEN_PICKLE = 'token.pickle' # Plik przechowujący token dostępu (zmieniono z token.json dla pickle)
CREDENTIALS_FILE = 'credentials.json'

# Ustawienia z pliku .env
GMB_ACCOUNT_ID = os.getenv('GMB_ACCOUNT_ID')
GMB_LOCATION_ID = os.getenv('GMB_LOCATION_ID')
SMTP_SERVER = os.getenv('SMTP_SERVER')
SMTP_PORT = int(os.getenv('SMTP_PORT', 587))
SMTP_USER = os.getenv('SMTP_USER')
SMTP_PASSWORD = os.getenv('SMTP_PASSWORD')
MANAGER_EMAIL = os.getenv('MANAGER_EMAIL')
NEGATIVE_THRESHOLD_STARS = int(os.getenv('NEGATIVE_THRESHOLD_STARS', 2))
POSITIVE_THRESHOLD_STARS = int(os.getenv('POSITIVE_THRESHOLD_STARS', 4))
SEND_DAILY_REPORT = os.getenv('SEND_DAILY_REPORT', 'true').lower() == 'true'
DAILY_REPORT_TIME = os.getenv('DAILY_REPORT_TIME', '09:00')
SEND_ALL_NOTIFICATIONS = os.getenv('SEND_ALL_NOTIFICATIONS', 'false').lower() == 'true'
AUTO_REPLY_TO_NEUTRAL = os.getenv('AUTO_REPLY_TO_NEUTRAL', 'false').lower() == 'true'
CHECK_INTERVAL_MINUTES = int(os.getenv('CHECK_INTERVAL_MINUTES', 60))
SENTIMENT_CONFIDENCE_THRESHOLD = float(os.getenv('SENTIMENT_CONFIDENCE_THRESHOLD', 0.75))
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')
COMPANY_NAME = os.getenv('COMPANY_NAME', 'naszej firmy')  # Domyślna wartość, jeśli nie ma w .env
COMPANY_CONTACT_INFO = os.getenv('COMPANY_CONTACT_INFO', '[Prosimy o kontakt bezpośredni]')  # Domyślna wartość

# --- Konfiguracja Gemini ---
gemini_model = None
if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        # Wybierz model - 'gemini-pro' jest dobrym ogólnym wyborem
        gemini_model = genai.GenerativeModel(
            model_name="gemini-pro",  # lub np. 'gemini-1.5-flash-latest'
            # Ustawienia bezpieczeństwa są ważne!
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        logging.info("Konfiguracja Google Gemini AI zakończona pomyślnie.")
    except Exception as e:
        logging.error(f"Błąd konfiguracji Gemini API: {e}. Sugestie AI nie będą dostępne.", exc_info=True)
        gemini_model = None
else:
    logging.warning("Brak klucza GEMINI_API_KEY w pliku .env. Sugestie AI nie będą dostępne.")

# Baza danych do śledzenia przetworzonych recenzji
DB_FILE = 'processed_reviews.db'

# Szablony odpowiedzi na pozytywne recenzje
POSITIVE_REPLY_TEMPLATES = [
    "Dziękujemy za pozytywną opinię! Cieszymy się, że wizyta była udana.",
    "Bardzo dziękujemy za miłe słowa! Zapraszamy ponownie.",
    "Dziękujemy za poświęcony czas i wysoką ocenę! To dla nas bardzo ważne.",
    "Super, że się podobało! Dziękujemy za 5 gwiazdek!",
    "Doceniamy Twoją opinię! Dziękujemy i pozdrawiamy!",
]

# Szablony odpowiedzi na neutralne recenzje
NEUTRAL_REPLY_TEMPLATES = [
    "Dziękujemy za Twoją opinię. Doceniamy każdą informację zwrotną, która pomaga nam się doskonalić.",
    "Dziękujemy za podzielenie się swoimi doświadczeniami. Jeśli masz jakieś sugestie, jak możemy poprawić naszą usługę, prosimy o kontakt.",
    "Cenimy Twoją opinię i zawsze dążymy do zapewnienia najlepszych doświadczeń. Dziękujemy za informację zwrotną."
]

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("bot.log"), logging.StreamHandler()])

# --- Funkcje Pomocnicze ---

def setup_database():
    """Inicjalizuje bazę danych SQLite, jeśli nie istnieje."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS processed_reviews (
        review_id TEXT PRIMARY KEY,
        processed_at TIMESTAMP,
        sentiment TEXT,
        stars INTEGER
    )
    ''')
    conn.commit()
    conn.close()
    logging.info("Baza danych gotowa.")

def is_review_processed(review_id: str) -> bool:
    """Sprawdza, czy ID recenzji znajduje się już w bazie danych."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT 1 FROM processed_reviews WHERE review_id = ?", (review_id,))
    result = cursor.fetchone()
    conn.close()
    return result is not None

def mark_review_as_processed(review_id: str, sentiment: str = 'UNKNOWN', stars: int = 0):
    """Dodaje ID recenzji do bazy danych wraz z dodatkowymi informacjami."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    try:
        cursor.execute(
            "INSERT INTO processed_reviews (review_id, processed_at, sentiment, stars) VALUES (?, ?, ?, ?)",
            (review_id, datetime.now(), sentiment, stars)
        )
        conn.commit()
        logging.info(f"Oznaczono recenzję {review_id} jako przetworzoną (sentyment: {sentiment}, gwiazdki: {stars}).")
    except sqlite3.IntegrityError:
        logging.warning(f"Recenzja {review_id} już istniała w bazie (próba ponownego oznaczenia).")
    except Exception as e:
        logging.error(f"Błąd podczas oznaczania recenzji {review_id} jako przetworzonej: {e}")
        conn.rollback() # Wycofaj zmiany w razie błędu
    finally:
        conn.close()


def authenticate_google_api():
    """Uwierzytelnia użytkownika i zwraca obiekt usługi GMB API."""
    creds = None
    # Plik token.pickle przechowuje tokeny dostępu i odświeżania użytkownika.
    if os.path.exists(TOKEN_PICKLE):
        with open(TOKEN_PICKLE, 'rb') as token_file:
            creds = pickle.load(token_file)

    # Jeśli nie ma ważnych danych uwierzytelniających, pozwól użytkownikowi się zalogować.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
            except Exception as e:
                logging.error(f"Błąd odświeżania tokena: {e}. Usuń {TOKEN_PICKLE} i spróbuj ponownie.")
                # W razie problemów z odświeżeniem, wymuś ponowną autoryzację
                if os.path.exists(TOKEN_PICKLE):
                    os.remove(TOKEN_PICKLE)
                # Uruchom ponownie przepływ autoryzacji
                flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
                creds = flow.run_local_server(port=0)
        else:
            flow = InstalledAppFlow.from_client_secrets_file(CREDENTIALS_FILE, SCOPES)
            creds = flow.run_local_server(port=0) # Uruchamia lokalny serwer do obsługi OAuth

        # Zapisz dane uwierzytelniające na następny raz
        with open(TOKEN_PICKLE, 'wb') as token_file:
            pickle.dump(creds, token_file)
        logging.info("Pomyślnie uzyskano i zapisano nowe dane uwierzytelniające.")

    try:
        # Budujemy dwa obiekty serwisów - jeden dla recenzji, drugi może być potrzebny dla info
        service_reviews = build(API_SERVICE_REVIEWS, API_VERSION_REVIEWS, credentials=creds)
        # service_info = build(API_SERVICE_NAME, API_VERSION_INFO, credentials=creds) # Jeśli potrzebne
        logging.info("Pomyślnie połączono z Google My Business API.")
        return service_reviews # , service_info
    except HttpError as error:
        logging.error(f'Wystąpił błąd podczas budowania usługi Google API: {error}')
        return None


def get_star_rating(rating_str: str) -> int:
    """Konwertuje string oceny (np. 'FIVE_STAR') na liczbę."""
    ratings = {'ONE_STAR': 1, 'TWO_STARS': 2, 'THREE_STARS': 3, 'FOUR_STARS': 4, 'FIVE_STARS': 5}
    # Poprawka dla różnych formatów API (czasem jest 'STARS_FIVE')
    ratings_alt = {'STAR_RATING_UNSPECIFIED': 0, 'ONE': 1, 'TWO': 2, 'THREE': 3, 'FOUR': 4, 'FIVE': 5}
    rating_str_upper = rating_str.upper()

    if rating_str_upper in ratings:
        return ratings[rating_str_upper]
    elif rating_str_upper in ratings_alt:
         return ratings_alt[rating_str_upper]
    # Próba bezpośredniej konwersji, jeśli API zwróciło liczbę
    try:
        return int(rating_str)
    except (ValueError, TypeError):
        logging.warning(f"Nieznany format oceny: {rating_str}")
        return 0 # Zwraca 0 jeśli nie można sparsować


# Typowanie dla wyniku analizy
class SentimentAnalysisResult(TypedDict):
    star_rating: int
    star_category: Literal['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'UNKNOWN']
    language: str | None
    text_sentiment: Literal['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'UNKNOWN', 'N/A'] # N/A jeśli brak tekstu
    decision: Literal['NOTIFY_MANAGER', 'AUTO_REPLY', 'IGNORE', 'NEEDS_MANUAL_REVIEW'] # Sugerowana akcja


def detect_review_language(text: str) -> str | None:
    """
    Wykrywa język podanego tekstu.

    Args:
        text: Tekst recenzji do analizy.

    Returns:
        Dwuliterowy kod języka (np. 'pl', 'en') lub None, jeśli tekst jest pusty
        lub wystąpił błąd podczas detekcji.
    """
    if not text or not isinstance(text, str) or not text.strip():
        logging.debug("Pusty tekst, nie można wykryć języka.")
        return None # Zwraca None dla pustego tekstu

    try:
        # Użyj bloku try-except, ponieważ langdetect może rzucić wyjątek dla bardzo krótkich/niejednoznacznych tekstów
        language_code = detect(text)
        logging.debug(f"Wykryty język: {language_code}")
        return language_code
    except LangDetectException as e:
        logging.warning(f"Nie udało się wykryć języka dla tekstu: '{text[:50]}...'. Błąd: {e}")
        return None # Zwraca None w przypadku błędu detekcji
    except Exception as e:
        logging.error(f"Nieoczekiwany błąd podczas wykrywania języka: {e}")
        return None


def _initialize_sentiment_pipeline():
    """Funkcja pomocnicza do inicjalizacji pipeline'u (leniwa inicjalizacja)"""
    global sentiment_pipeline
    if sentiment_pipeline is None:
        try:
            logging.info(f"Inicjalizowanie potoku analizy sentymentu dla modelu: {MODEL_NAME}")
            # Jawne ładowanie modelu i tokenizera może dać większą kontrolę
            model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model=model,
                tokenizer=tokenizer
                # Można dodać device=0 jeśli masz GPU i chcesz go użyć (wymaga CUDA)
            )
            logging.info("Potok analizy sentymentu zainicjalizowany pomyślnie.")
        except Exception as e:
            logging.error(f"Nie udało się załadować modelu sentymentu {MODEL_NAME}: {e}", exc_info=True)
            # W przypadku błędu, pipeline pozostanie None, analiza tekstu nie zadziała


def analyze_review_advanced(
    review: dict,
    negative_threshold_stars: int = NEGATIVE_THRESHOLD_STARS,
    positive_threshold_stars: int = POSITIVE_THRESHOLD_STARS,
    sentiment_confidence_threshold: float = SENTIMENT_CONFIDENCE_THRESHOLD
    ) -> SentimentAnalysisResult:
    """
    Zaawansowana analiza recenzji, uwzględniająca gwiazdki, język i sentyment tekstu.
    Określa sugerowaną akcję (powiadomienie, odpowiedź, ignorowanie).

    Args:
        review: Słownik z danymi recenzji (oczekuje kluczy 'starRating', 'comment').
        negative_threshold_stars: Próg dla negatywnej oceny gwiazdkowej (<=).
        positive_threshold_stars: Próg dla pozytywnej oceny gwiazdkowej (>=).
        sentiment_confidence_threshold: Minimalna pewność modelu, by uznać sentyment tekstu.

    Returns:
        Słownik SentimentAnalysisResult z wynikami analizy i sugerowaną decyzją.
    """
    # Inicjalizuj pipeline jeśli jeszcze nie jest gotowy
    if sentiment_pipeline is None:
        _initialize_sentiment_pipeline()

    # 1. Analiza gwiazdek
    star_rating_str = review.get('starRating', review.get('rating', 'STAR_RATING_UNSPECIFIED'))
    stars = get_star_rating(star_rating_str)
    comment = review.get('comment', '')

    logging.info(f"Analizuję recenzję: Gwiazdki={stars}, Komentarz='{comment[:50]}...'")

    star_category: Literal['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'UNKNOWN'] = 'UNKNOWN'
    if stars == 0:
         star_category = 'UNKNOWN'
    elif stars <= negative_threshold_stars:
        star_category = 'NEGATIVE'
    elif stars >= positive_threshold_stars:
        star_category = 'POSITIVE'
    else:
        star_category = 'NEUTRAL' # Np. 3 gwiazdki

    # 2. Analiza tekstu (jeśli istnieje)
    language: str | None = None
    text_sentiment: Literal['POSITIVE', 'NEUTRAL', 'NEGATIVE', 'UNKNOWN', 'N/A'] = 'N/A' # Domyślnie N/A jeśli brak komentarza

    if comment and isinstance(comment, str) and comment.strip():
        language = detect_review_language(comment) # Użyj modułu detekcji języka

        # Jeśli wykryto polski i model jest dostępny
        if language == 'pl' and sentiment_pipeline:
            try:
                # Uruchom analizę sentymentu
                results = sentiment_pipeline(comment)
                # Wynik to zwykle lista, bierzemy pierwszy element
                if results:
                    result = results[0]
                    label = result.get('label')
                    score = result.get('score')
                    logging.debug(f"Analiza tekstu: label={label}, score={score:.4f}")

                    # Klasyfikacja sentymentu tekstu na podstawie etykiety i pewności
                    if label == 'negative' and score >= sentiment_confidence_threshold:
                        text_sentiment = 'NEGATIVE'
                    elif label == 'positive' and score >= sentiment_confidence_threshold:
                        text_sentiment = 'POSITIVE'
                    else:
                        # Jeśli pewność jest niska lub etykieta inna (np. neutralna, jeśli model ją zwraca)
                        text_sentiment = 'NEUTRAL'
                else:
                    text_sentiment = 'UNKNOWN' # Jeśli pipeline nic nie zwrócił
            except Exception as e:
                logging.error(f"Błąd podczas analizy sentymentu tekstu: {e}", exc_info=True)
                text_sentiment = 'UNKNOWN'
        elif language != 'pl':
            logging.info(f"Wykryto język '{language}', pomijam analizę sentymentu tekstu (model polski).")
            text_sentiment = 'UNKNOWN' # Nie mamy modelu dla tego języka
        else: # language == 'pl' ale sentiment_pipeline is None (błąd ładowania)
             logging.warning("Nie można przeprowadzić analizy sentymentu tekstu - model nie został załadowany.")
             text_sentiment = 'UNKNOWN'

    else: # Brak komentarza
        text_sentiment = 'N/A'

    # 3. Logika decyzyjna - kluczowy element nowej logiki
    decision: Literal['NOTIFY_MANAGER', 'AUTO_REPLY', 'IGNORE', 'NEEDS_MANUAL_REVIEW']

    if star_category == 'NEGATIVE' or text_sentiment == 'NEGATIVE':
        # ZAWSZE powiadamiaj jeśli gwiazdki są niskie LUB tekst jest negatywny (nawet przy dobrych gwiazdkach!)
        decision = 'NOTIFY_MANAGER'
        logging.info(f"Decyzja dla recenzji {review.get('reviewId')}: NOTIFY_MANAGER (stars={stars}, text={text_sentiment})")
    elif star_category == 'POSITIVE' and text_sentiment != 'NEGATIVE':
        # Odpowiadaj automatycznie TYLKO jeśli gwiazdki są wysokie ORAZ tekst NIE jest negatywny
        # Dodatkowy warunek: odpowiadaj tylko na polskie komentarze (jeśli chcesz)
        if language == 'pl' and text_sentiment != 'N/A': # Sprawdzamy czy jest komentarz (text_sentiment != 'N/A')
             decision = 'AUTO_REPLY'
             logging.info(f"Decyzja dla recenzji {review.get('reviewId')}: AUTO_REPLY (stars={stars}, text={text_sentiment}, lang={language})")
        else:
             # Pozytywne gwiazdki, nie-negatywny tekst, ale inny język lub brak komentarza - można zignorować lub oznaczyć do ręcznego przeglądu
             decision = 'IGNORE' # Lub 'NEEDS_MANUAL_REVIEW' jeśli chcesz przeglądać nie-polskie
             logging.info(f"Decyzja dla recenzji {review.get('reviewId')}: IGNORE (pozytywne gwiazdki, brak neg. tekstu, język={language} lub brak tekstu)")
    else: # Neutralne gwiazdki (np. 3) i nie-negatywny tekst, lub nieznany sentyment tekstu
        decision = 'NEEDS_MANUAL_REVIEW' # Lepiej przejrzeć ręcznie niż ignorować lub źle odpowiadać
        logging.info(f"Decyzja dla recenzji {review.get('reviewId')}: NEEDS_MANUAL_REVIEW (stars={stars}, text={text_sentiment}, lang={language})")

    return {
        'star_rating': stars,
        'star_category': star_category,
        'language': language,
        'text_sentiment': text_sentiment,
        'decision': decision
    }


def analyze_sentiment(review: dict) -> str:
    """
    Stara funkcja analizy sentymentu (zachowana dla kompatybilności).
    Zalecane jest używanie analyze_review_advanced.
    """
    # Wywołaj nową funkcję i zwróć tylko kategorię sentymentu
    result = analyze_review_advanced(review)
    return result['star_category']


def generate_gemini_prompt_for_negative_review(review: dict) -> str:
    """Tworzy prompt dla Gemini do generowania odpowiedzi na negatywną opinię."""
    stars = get_star_rating(review.get('starRating', '0'))
    # Obsługa anonimowego autora lub "Użytkownik Google"
    raw_author = review.get('reviewer', {}).get('displayName', 'Klient')
    author = raw_author if raw_author != "Użytkownik Google" else "Klient"
    comment = review.get('comment', 'Brak komentarza.')

    prompt = f"""
Jesteś asystentem AI pomagającym managerowi {COMPANY_NAME} w tworzeniu odpowiedzi na negatywne opinie Google.
Twoim zadaniem jest przygotowanie szkicu profesjonalnej, grzecznej, empatycznej i konstruktywnej odpowiedzi na poniższą recenzję, zgodnie z najlepszymi praktykami obsługi klienta.

Otrzymaliśmy następującą negatywną opinię:
Ocena: {stars} / 5 gwiazdek
Autor: {author}
Treść: "{comment}"

Wytyczne do stworzenia odpowiedzi:
1.  Rozpocznij od podziękowania za opinię (nawet jeśli jest negatywna).
2.  Jeśli autor podał imię ({author}, jeśli różne od "Klient"), użyj go w powitaniu (np. "Panie/Pani [Imię]," lub "Szanowny/Szanowna [Imię],").
3.  Wyraź zrozumienie lub ubolewanie z powodu negatywnego doświadczenia opisanego przez klienta. Okaż empatię.
4.  Przeproś za niedogodności lub za to, że doświadczenie nie spełniło oczekiwań (unikaj bezpośredniego przyznawania się do winy, jeśli nie znasz faktów, skup się na odczuciach klienta).
5.  Jeśli to możliwe i bezpieczne, odnieś się bardzo krótko do głównego problemu poruszonego w recenzji, pokazując, że została przeczytana. Nie wdawaj się w publiczne spory ani szczegółowe wyjaśnienia.
6.  Zaproponuj przeniesienie rozmowy do kanału prywatnego w celu dokładniejszego wyjaśnienia sprawy i znalezienia rozwiązania. Podaj dane kontaktowe: {COMPANY_CONTACT_INFO}.
7.  Zakończ profesjonalnym pozdrowieniem.
8.  Odpowiedź musi być w języku polskim.
9.  Zachowaj uprzejmy, profesjonalny i spokojny ton.
10. Odpowiedź powinna być zwięzła.

Przygotuj tylko tekst samej odpowiedzi, bez żadnych dodatkowych wyjaśnień z Twojej strony.
"""
    return prompt.strip()


def get_gemini_suggestion(prompt: str) -> str | None:
    """Wysyła prompt do Gemini API i zwraca sugerowaną odpowiedź."""
    if not gemini_model:
        logging.warning("Model Gemini nie jest dostępny. Nie można wygenerować sugestii.")
        return None

    try:
        # Ustawienia generowania (opcjonalne, dostosuj wg potrzeb)
        generation_config = genai.types.GenerationConfig(
            # candidate_count=1, # Zwykle chcemy jedną najlepszą sugestię
            # stop_sequences=['\n\n'], # Można ustawić sekwencje zatrzymujące generowanie
            max_output_tokens=512,  # Limit długości odpowiedzi
            temperature=0.7,  # Kreatywność vs Spójność (0.0 - bardzo spójne, 1.0 - bardzo kreatywne)
            # top_p=0.9, # Inna metoda próbkowania
            # top_k=40  # Jeszcze inna metoda próbkowania
        )

        logging.info("Wysyłanie promptu do Gemini API...")
        response = gemini_model.generate_content(
            prompt,
            generation_config=generation_config
        )

        # Sprawdzenie, czy odpowiedź nie została zablokowana przez filtry bezpieczeństwa
        if response.prompt_feedback and response.prompt_feedback.block_reason:
            logging.error(f"Prompt został zablokowany przez filtry bezpieczeństwa Gemini. Powód: {response.prompt_feedback.block_reason}")
            return f"[Błąd: Prompt zablokowany przez filtry bezpieczeństwa Gemini ({response.prompt_feedback.block_reason})]"

        # Sprawdzenie czy są kandydaci i tekst
        if response.candidates and response.candidates[0].content.parts:
            suggestion = response.text  # .text jest wygodnym skrótem do pobrania tekstu
            logging.info("Otrzymano sugestię odpowiedzi od Gemini.")
            # Proste czyszczenie - usuwanie ewentualnych pustych linii na początku/końcu
            return suggestion.strip()
        else:
            logging.warning("Gemini API nie zwróciło tekstu w odpowiedzi.")
            # Sprawdzenie przyczyny w bardziej złożonych przypadkach
            logging.debug(f"Pełna odpowiedź Gemini: {response}")
            # Zwróć informację o braku odpowiedzi zamiast None
            return "[Informacja: Gemini nie wygenerowało treści dla tego promptu]"

    except google_exceptions.GoogleAPIError as e:
        logging.error(f"Błąd API Google podczas komunikacji z Gemini: {e}", exc_info=True)
        return f"[Błąd: Problem z połączeniem z Gemini API ({e})]"
    except Exception as e:
        logging.error(f"Nieoczekiwany błąd podczas generowania sugestii przez Gemini: {e}", exc_info=True)
        return f"[Błąd: Wewnętrzny błąd podczas generowania sugestii ({e})]"


def send_notification_email(review: dict, sentiment: str = 'NEGATIVE', gemini_suggestion: str | None = None):
    """Wysyła email z powiadomieniem o recenzji, opcjonalnie z sugestią AI."""
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, MANAGER_EMAIL]):
        logging.error("Brak pełnej konfiguracji SMTP w pliku .env. Nie można wysłać emaila.")
        return

    review_id = review.get('reviewId', 'Brak ID')
    reviewer_name = review.get('reviewer', {}).get('displayName', 'Anonim')
    stars = get_star_rating(review.get('starRating', '0'))
    comment = review.get('comment', 'Brak komentarza.')
    create_time = review.get('createTime', 'Brak daty')
    review_name = review.get('name') # Format: accounts/{accountId}/locations/{locationId}/reviews/{reviewId}

    # Link do bezpośredniej odpowiedzi w panelu Google (konstruowany)
    parent_location = '/'.join(review_name.split('/')[:-2]) if review_name else GMB_ACCOUNT_ID + '/' + GMB_LOCATION_ID if GMB_ACCOUNT_ID and GMB_LOCATION_ID else None
    google_review_link = f"https://business.google.com/reviews/l/{parent_location.split('/')[-1]}" if parent_location else "Link niedostępny"

    # Dostosowanie tematu i stylu wiadomości w zależności od sentymentu
    if sentiment == 'NEGATIVE':
        subject = f"❗️ Nowa Negatywna Opinia Google ({stars}★) od {reviewer_name}"
        priority = "Wysoki"
        color = "#FF4136" # Czerwony
    elif sentiment == 'POSITIVE':
        subject = f"✅ Nowa Pozytywna Opinia Google ({stars}★) od {reviewer_name}"
        priority = "Normalny"
        color = "#2ECC40" # Zielony
    else: # NEUTRAL
        subject = f"ℹ️ Nowa Neutralna Opinia Google ({stars}★) od {reviewer_name}"
        priority = "Normalny"
        color = "#FFDC00" # Żółty

    # Tworzenie wiadomości HTML
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .container {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .header {{ background-color: {color}; color: white; padding: 10px; border-radius: 5px 5px 0 0; }}
            .content {{ padding: 15px; }}
            .review-text {{ background-color: #f9f9f9; padding: 15px; border-left: 4px solid {color}; margin: 10px 0; }}
            .footer {{ font-size: 12px; color: #777; margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px; }}
            .stars {{ color: #FFD700; }} /* Złoty kolor dla gwiazdek */
            .info-row {{ margin-bottom: 10px; }}
            .button {{ background-color: {color}; color: white; padding: 10px 15px; text-decoration: none; border-radius: 4px; display: inline-block; margin-top: 15px; }}
            .ai-suggestion {{ background-color: #E8F4FD; padding: 15px; border-left: 4px solid #4285F4; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Nowa Opinia Google dla {COMPANY_NAME}</h2>
            </div>
            <div class="content">
                <div class="info-row"><strong>Autor:</strong> {reviewer_name}</div>
                <div class="info-row"><strong>Ocena:</strong> <span class="stars">{'★' * stars}{'☆' * (5-stars)}</span> ({stars}/5)</div>
                <div class="info-row"><strong>Data:</strong> {create_time}</div>
                <div class="info-row"><strong>Priorytet:</strong> {priority}</div>
                
                <h3>Treść opinii:</h3>
                <div class="review-text">
                    {comment}
                </div>
                
                <div class="info-row"><strong>ID Recenzji:</strong> {review_id}</div>
                <a href="{google_review_link}" class="button">Odpowiedz na opinię</a>
    """

    # Dodanie sekcji z sugestią AI, jeśli dostępna
    if gemini_suggestion:
        html_body += f"""
                <h3>🤖 Sugestia odpowiedzi wygenerowana przez AI (Gemini):</h3>
                <p><em>Pamiętaj, aby ją sprawdzić, dostosować i spersonalizować przed użyciem!</em></p>
                <div class="ai-suggestion">
                    {gemini_suggestion.replace('\n', '<br>')}
                </div>
        """
    elif sentiment == 'NEGATIVE':
        html_body += """
                <p><em>(Sugestia odpowiedzi AI nie jest dostępna - sprawdź logi bota lub konfigurację Gemini API)</em></p>
        """

    # Dokończenie HTML
    html_body += """
            </div>
            <div class="footer">
                <p>Wiadomość wygenerowana automatycznie przez Bota Analizującego Opinie Google.</p>
                <p>Jeśli nie chcesz otrzymywać tych powiadomień, zmień ustawienia w pliku konfiguracyjnym.</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Tworzenie wiadomości tekstowej (alternatywa dla klientów bez obsługi HTML)
    text_body = f"""
    Otrzymano nową opinię w Google Moja Firma dla {COMPANY_NAME}:

    Autor: {reviewer_name}
    Ocena: {stars} / 5 ★
    Data: {create_time}
    Priorytet: {priority}
    
    Treść:
    --------------------
    {comment}
    --------------------

    ID Recenzji: {review_id}
    
    Proszę o weryfikację i odpowiedź.
    Link do panelu opinii: {google_review_link}
    """

    # Dodanie sekcji z sugestią AI do wersji tekstowej, jeśli dostępna
    if gemini_suggestion:
        text_body += f"""

🤖 Sugestia odpowiedzi wygenerowana przez AI (Gemini):
   (Pamiętaj, aby ją sprawdzić, dostosować i spersonalizować przed użyciem!)
--------------------
{gemini_suggestion}
--------------------
"""
    elif sentiment == 'NEGATIVE':
        text_body += """

(Sugestia odpowiedzi AI nie jest dostępna - sprawdź logi bota lub konfigurację Gemini API)
"""

    text_body += """
---
Wiadomość wygenerowana automatycznie przez Bota Analizującego Opinie.
"""

    # Tworzenie wiadomości multipart (HTML + tekst)
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = f"{COMPANY_NAME} Bot <{SMTP_USER}>"  # Dodanie nazwy firmy w polu Od
    msg['To'] = MANAGER_EMAIL
    
    # Dodawanie części tekstowej i HTML
    part1 = MIMEText(text_body, 'plain', 'utf-8')
    part2 = MIMEText(html_body, 'html', 'utf-8')
    
    # Dodawanie części do wiadomości (tekst jako pierwszy, HTML jako drugi)
    msg.attach(part1)
    msg.attach(part2)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo() # Przywitanie z serwerem
            server.starttls() # Uruchomienie szyfrowania TLS
            server.ehlo() # Ponowne przywitanie po TLS
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [MANAGER_EMAIL], msg.as_string())
        logging.info(f"Wysłano powiadomienie email o recenzji {review_id} (sentyment: {sentiment}) do {MANAGER_EMAIL} (z sugestią AI: {'Tak' if gemini_suggestion else 'Nie'}).")
    except smtplib.SMTPAuthenticationError:
        logging.error(f"Błąd autentykacji SMTP dla użytkownika {SMTP_USER}. Sprawdź login/hasło.")
    except Exception as e:
        logging.error(f"Nie udało się wysłać emaila o recenzji {review_id}: {e}")


def generate_reply_text(review: dict, sentiment: str = 'POSITIVE') -> str:
    """Generuje spersonalizowaną odpowiedź na recenzję w zależności od sentymentu."""
    reviewer_name = review.get('reviewer', {}).get('displayName')
    first_name = reviewer_name.split()[0] if reviewer_name and reviewer_name != "Użytkownik Google" else None
    
    # Wybór szablonu w zależności od sentymentu
    if sentiment == 'POSITIVE':
        template = random.choice(POSITIVE_REPLY_TEMPLATES)
    elif sentiment == 'NEUTRAL':
        template = random.choice(NEUTRAL_REPLY_TEMPLATES)
    else:
        # Dla negatywnych recenzji nie generujemy automatycznej odpowiedzi
        return None
    
    # Personalizacja odpowiedzi
    current_hour = datetime.now().hour
    time_greeting = ""
    if 5 <= current_hour < 12:
        time_greeting = "Dzień dobry! "
    elif 12 <= current_hour < 18:
        time_greeting = "Witamy! "
    elif 18 <= current_hour < 22:
        time_greeting = "Dobry wieczór! "
    
    # Dodanie imienia i powitania
    if first_name:
        return f"{time_greeting}Dziękujemy za opinię, {first_name}! {template}"
    else:
        return f"{time_greeting}{template}"


def post_google_reply(service, review_name: str, reply_text: str):
    """Publikuje odpowiedź na recenzję za pomocą GMB API."""
    # review_name to pełny identyfikator recenzji: accounts/{accountId}/locations/{locationId}/reviews/{reviewId}
    if not review_name:
        logging.error("Brak 'name' recenzji, nie można opublikować odpowiedzi.")
        return False

    # Sprawdzenie, czy recenzja ma już odpowiedź (ważne, by nie nadpisywać ręcznych odpowiedzi!)
    try:
        review_data = service.accounts().locations().reviews().get(name=review_name).execute()
        if review_data.get('reviewReply'):
            logging.warning(f"Recenzja {review_name} ma już odpowiedź. Pomijam automatyczną odpowiedź.")
            return False # Zwracamy False, bo nie wykonaliśmy akcji, ale nie jest to błąd krytyczny
    except HttpError as error:
         logging.error(f'Błąd podczas sprawdzania odpowiedzi dla recenzji {review_name}: {error}')
         # Mimo błędu, spróbujmy odpowiedzieć, ale ostrożnie
         pass # Kontynuuj z próbą odpowiedzi

    # Publikowanie odpowiedzi
    reply_body = {'comment': reply_text}
    try:
        request = service.accounts().locations().reviews().updateReply(
            name=review_name,
            body=reply_body
        )
        response = request.execute()
        logging.info(f"Pomyślnie opublikowano odpowiedź na recenzję {review_name}: {response}")
        return True
    except HttpError as error:
        logging.error(f"Błąd podczas publikowania odpowiedzi na recenzję {review_name}: {error}")
        # Sprawdzenie konkretnych błędów, np. 'ALREADY_EXISTS' jeśli jednak była odpowiedź
        if 'ALREADY_EXISTS' in str(error):
             logging.warning(f"Recenzja {review_name} miała już odpowiedź (błąd ALREADY_EXISTS). Oznaczam jako OK.")
             return False # Traktujemy jako sytuację, gdzie nie trzeba już odpowiadać
        return False # Zwracamy False przy innych błędach API


def send_daily_report():
    """Wysyła dzienny raport podsumowujący aktywność recenzji."""
    if not all([SMTP_SERVER, SMTP_PORT, SMTP_USER, SMTP_PASSWORD, MANAGER_EMAIL]):
        logging.error("Brak pełnej konfiguracji SMTP w pliku .env. Nie można wysłać raportu dziennego.")
        return
    
    # Pobieranie statystyk z ostatnich 24 godzin
    yesterday = datetime.now() - timedelta(days=1)
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    
    # Liczba przetworzonych recenzji w ciągu ostatnich 24h
    cursor.execute("SELECT COUNT(*) FROM processed_reviews WHERE processed_at >= ?", 
                  (yesterday.strftime('%Y-%m-%d %H:%M:%S'),))
    processed_count = cursor.fetchone()[0]
    
    # Statystyki według sentymentu
    cursor.execute("SELECT sentiment, COUNT(*) FROM processed_reviews WHERE processed_at >= ? GROUP BY sentiment", 
                  (yesterday.strftime('%Y-%m-%d %H:%M:%S'),))
    sentiment_stats = cursor.fetchall()
    
    # Statystyki według gwiazdek
    cursor.execute("SELECT stars, COUNT(*) FROM processed_reviews WHERE processed_at >= ? GROUP BY stars ORDER BY stars DESC", 
                  (yesterday.strftime('%Y-%m-%d %H:%M:%S'),))
    star_stats = cursor.fetchall()
    
    conn.close()
    
    # Przygotowanie danych do raportu
    sentiment_data = {sentiment: count for sentiment, count in sentiment_stats}
    positive_count = sentiment_data.get('POSITIVE', 0)
    negative_count = sentiment_data.get('NEGATIVE', 0)
    neutral_count = sentiment_data.get('NEUTRAL', 0)
    
    # Tworzenie raportu HTML
    subject = f"📊 Dzienny Raport Recenzji Google - {datetime.now().strftime('%d.%m.%Y')}"
    
    html_body = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
            .container {{ padding: 20px; border: 1px solid #ddd; border-radius: 5px; }}
            .header {{ background-color: #4285F4; color: white; padding: 10px; border-radius: 5px 5px 0 0; }}
            .content {{ padding: 15px; }}
            .stat-box {{ background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }}
            .footer {{ font-size: 12px; color: #777; margin-top: 20px; border-top: 1px solid #eee; padding-top: 10px; }}
            .button {{ background-color: #4285F4; color: white; padding: 10px 15px; text-decoration: none; 
                      border-radius: 4px; display: inline-block; margin-top: 15px; }}
            .stars {{ color: #FFD700; }} /* Złoty kolor dla gwiazdek */
            .stat-row {{ display: flex; justify-content: space-between; margin-bottom: 5px; }}
            .stat-label {{ font-weight: bold; }}
            .positive {{ color: #2ECC40; }}
            .negative {{ color: #FF4136; }}
            .neutral {{ color: #FFDC00; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Dzienny Raport Recenzji Google</h2>
            </div>
            <div class="content">
                <h3>Podsumowanie ostatnich 24 godzin</h3>
                
                <div class="stat-box">
                    <div class="stat-row">
                        <span class="stat-label">Liczba przetworzonych recenzji:</span>
                        <span><strong>{processed_count}</strong></span>
                    </div>
                    
                    <h4>Według sentymentu:</h4>
                    <div class="stat-row">
                        <span class="stat-label">Pozytywne:</span>
                        <span class="positive"><strong>{positive_count}</strong></span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Neutralne:</span>
                        <span class="neutral"><strong>{neutral_count}</strong></span>
                    </div>
                    <div class="stat-row">
                        <span class="stat-label">Negatywne:</span>
                        <span class="negative"><strong>{negative_count}</strong></span>
                    </div>
                    
                    <h4>Według gwiazdek:</h4>
                    {"".join([f'<div class="stat-row"><span class="stat-label">{stars} {"★" * stars}:</span><span><strong>{count}</strong></span></div>' for stars, count in star_stats])}
                </div>
                
                <p>Aby zobaczyć wszystkie recenzje, kliknij poniższy przycisk:</p>
                <a href="https://business.google.com/reviews" class="button">Przejdź do panelu recenzji</a>
            </div>
            <div class="footer">
                <p>Wiadomość wygenerowana automatycznie przez Bota Analizującego Opinie Google.</p>
                <p>Jeśli nie chcesz otrzymywać tych raportów, zmień ustawienie SEND_DAILY_REPORT w pliku konfiguracyjnym.</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Tworzenie wiadomości tekstowej (alternatywa dla klientów bez obsługi HTML)
    text_body = f"""
    DZIENNY RAPORT RECENZJI GOOGLE - {datetime.now().strftime('%d.%m.%Y')}
    
    Podsumowanie ostatnich 24 godzin:
    
    * Liczba przetworzonych recenzji: {processed_count}
    
    Według sentymentu:
    * Pozytywne: {positive_count}
    * Neutralne: {neutral_count}
    * Negatywne: {negative_count}
    
    Według gwiazdek:
    {chr(10).join([f'* {stars} gwiazdek: {count}' for stars, count in star_stats])}
    
    Aby zobaczyć wszystkie recenzje, przejdź do:
    https://business.google.com/reviews
    
    ---
    Wiadomość wygenerowana automatycznie przez Bota Analizującego Opinie Google.
    Jeśli nie chcesz otrzymywać tych raportów, zmień ustawienie SEND_DAILY_REPORT w pliku konfiguracyjnym.
    """
    
    # Tworzenie wiadomości multipart (HTML + tekst)
    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = SMTP_USER
    msg['To'] = MANAGER_EMAIL
    
    # Dodawanie części tekstowej i HTML
    part1 = MIMEText(text_body, 'plain', 'utf-8')
    part2 = MIMEText(html_body, 'html', 'utf-8')
    
    # Dodawanie części do wiadomości (tekst jako pierwszy, HTML jako drugi)
    msg.attach(part1)
    msg.attach(part2)
    
    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.ehlo()
            server.starttls()
            server.ehlo()
            server.login(SMTP_USER, SMTP_PASSWORD)
            server.sendmail(SMTP_USER, [MANAGER_EMAIL], msg.as_string())
        logging.info(f"Wysłano dzienny raport recenzji do {MANAGER_EMAIL}.")
    except Exception as e:
        logging.error(f"Nie udało się wysłać dziennego raportu: {e}")


def schedule_daily_tasks():
    """Ustawia harmonogram zadań cyklicznych."""
    if SEND_DAILY_REPORT:
        schedule.every().day.at(DAILY_REPORT_TIME).do(send_daily_report)
        logging.info(f"Zaplanowano wysyłanie dziennego raportu o godzinie {DAILY_REPORT_TIME}.")
    
    # Planowanie regularnego sprawdzania recenzji
    schedule.every(CHECK_INTERVAL_MINUTES).minutes.do(process_reviews)
    logging.info(f"Zaplanowano sprawdzanie recenzji co {CHECK_INTERVAL_MINUTES} minut.")


def run_scheduler():
    """Uruchamia scheduler w osobnym wątku."""
    while True:
        schedule.run_pending()
        time.sleep(60)  # Sprawdzaj co minutę


def process_reviews():
    """Główna funkcja przetwarzająca recenzje."""
    logging.info("--- Rozpoczęcie przetwarzania recenzji ---")
    service_reviews = authenticate_google_api()

    if not service_reviews:
        logging.critical("Nie udało się uwierzytelnić w Google API. Zakończenie pracy.")
        return

    if not GMB_ACCOUNT_ID or not GMB_LOCATION_ID:
        logging.critical("Brak GMB_ACCOUNT_ID lub GMB_LOCATION_ID w konfiguracji (.env). Zakończenie pracy.")
        return

    parent = f"{GMB_ACCOUNT_ID}/{GMB_LOCATION_ID}"
    processed_count = 0
    negative_count = 0
    positive_count = 0
    neutral_count = 0
    manual_review_count = 0
    replied_count = 0

    try:
        # Pobieranie recenzji - używamy reviews.list
        request = service_reviews.accounts().locations().reviews().list(
            parent=parent,
            pageSize=50,  # Pobierz do 50 recenzji na raz
            orderBy="updateTime desc"  # Sortuj od najnowszych
        )
        response = request.execute()
        reviews = response.get('reviews', [])

        logging.info(f"Pobrano {len(reviews)} recenzji dla lokalizacji {parent}.")

        for review in reviews:
            review_id = review.get('reviewId')
            review_name = review.get('name')  # Pełny identyfikator

            if not review_id or not review_name:
                logging.warning(f"Pominięto recenzję bez ID lub Name: {review}")
                continue

            if is_review_processed(review_id):
                logging.debug(f"Recenzja {review_id} była już przetwarzana. Pomijam.")
                continue

            # Podstawowa walidacja recenzji
            star_rating_str = review.get('starRating', review.get('rating', 'STAR_RATING_UNSPECIFIED'))
            if not star_rating_str:
                logging.warning(f"Pominięto recenzję {review_id} bez oceny gwiazdkowej.")
                mark_review_as_processed(review_id)  # Oznacz jako przetworzoną, by nie wracać
                continue

            # Zaawansowana analiza recenzji
            analysis_result = analyze_review_advanced(review)
            stars = analysis_result['star_rating']
            sentiment = analysis_result['star_category']
            language = analysis_result['language']
            text_sentiment = analysis_result['text_sentiment']
            decision = analysis_result['decision']
            
            processed_count += 1

            # Zapisanie informacji o recenzji w bazie danych
            mark_review_as_processed(review_id, sentiment, stars)

            # Obsługa recenzji w zależności od decyzji
            if decision == 'NOTIFY_MANAGER':
                negative_count += 1
                
                # Generowanie sugestii odpowiedzi dla negatywnych recenzji za pomocą Gemini AI
                gemini_suggestion_text = None
                if gemini_model:
                    prompt = generate_gemini_prompt_for_negative_review(review)
                    gemini_suggestion_text = get_gemini_suggestion(prompt)
                
                # Wysyłanie powiadomienia z sugestią odpowiedzi
                send_notification_email(review, 'NEGATIVE', gemini_suggestion_text)
            
            elif decision == 'AUTO_REPLY':
                positive_count += 1
                # Wysyłanie powiadomień o pozytywnych recenzjach, jeśli włączone
                if SEND_ALL_NOTIFICATIONS:
                    send_notification_email(review, sentiment)
                
                # Automatyczna odpowiedź na pozytywne recenzje z komentarzem
                reply_text = generate_reply_text(review, sentiment)
                if post_google_reply(service_reviews, review_name, reply_text):
                    replied_count += 1
            
            elif decision == 'IGNORE':
                if sentiment == 'POSITIVE':
                    positive_count += 1
                elif sentiment == 'NEUTRAL':
                    neutral_count += 1
                
                # Wysyłanie powiadomień o wszystkich recenzjach, jeśli włączone
                if SEND_ALL_NOTIFICATIONS:
                    send_notification_email(review, sentiment)
            
            elif decision == 'NEEDS_MANUAL_REVIEW':
                manual_review_count += 1
                if sentiment == 'NEUTRAL':
                    neutral_count += 1
                
                # Wysyłanie powiadomień o recenzjach wymagających ręcznej weryfikacji
                send_notification_email(review, 'NEUTRAL')  # Wysyłamy jako neutralną dla ręcznej weryfikacji

        logging.info(f"--- Zakończono przetwarzanie recenzji ---")
        logging.info(f"Nowo przetworzonych recenzji: {processed_count}")
        logging.info(f"Pozytywnych: {positive_count}, Neutralnych: {neutral_count}, Negatywnych: {negative_count}")
        logging.info(f"Wymagających ręcznej weryfikacji: {manual_review_count}")
        logging.info(f"Automatycznie odpowiedziano na: {replied_count}")

    except HttpError as error:
        logging.error(f'Wystąpił błąd API Google podczas pobierania/przetwarzania recenzji: {error}')
    except Exception as e:
        logging.exception("Wystąpił nieoczekiwany błąd w głównej pętli bota:")


def main():
    """Główna funkcja uruchamiająca bota."""
    logging.info("=== Uruchomienie Bota Analizującego Opinie Google ===")
    setup_database()
    
    # Uruchomienie pierwszego przetwarzania recenzji
    process_reviews()
    
    # Konfiguracja harmonogramu zadań
    schedule_daily_tasks()
    
    # Uruchomienie schedulera w osobnym wątku
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.daemon = True  # Wątek zostanie zakończony, gdy główny program się zakończy
    scheduler_thread.start()
    
    try:
        # Utrzymanie głównego wątku przy życiu
        while True:
            time.sleep(3600)  # Sprawdzaj co godzinę, czy główny wątek działa
    except KeyboardInterrupt:
        logging.info("Otrzymano sygnał przerwania. Kończenie pracy bota.")
    except Exception as e:
        logging.exception(f"Wystąpił nieoczekiwany błąd w głównym wątku: {e}")
    finally:
        logging.info("=== Zakończenie pracy Bota Analizującego Opinie Google ===")


if __name__ == '__main__':
    main()
