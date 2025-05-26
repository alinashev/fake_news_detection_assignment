import json
import os
import re
import string

from bs4 import BeautifulSoup
from nltk import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import stopwords

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def load_limited_leaky_words(path="../registry/vocabularies/leaky_words.json", top_fake=10000, top_real=10000) -> set:
    """
    Loads a limited set of leaky words (tokens unique to fake or real news) from a JSON file.

    Args:
        path (str): Path to the leaky words JSON file.
        top_fake (int): Maximum number of fake-specific tokens to load.
        top_real (int): Maximum number of real-specific tokens to load.

    Returns:
        set: A set of leaky tokens from both fake and real news.
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fake_only = data.get("fake_only", [])[:top_fake]
        real_only = data.get("real_only", [])[:top_real]

        return set(fake_only + real_only)
    return set()


LEAKY_WORDS = load_limited_leaky_words(top_fake=100, top_real=100)


def remove_leaky_tokens(tokens: list) -> list:
    """
    Removes leaky tokens from the given list of tokens.

    Args:
        tokens (list of str): List of tokens to filter.

    Returns:
        list of str: Tokens with leaky words removed.
    """
    return [t for t in tokens if t not in LEAKY_WORDS]


def clean_text(text: str) -> str:
    """
    Cleans raw text by applying:
    - lowercasing
    - HTML removal
    - URL and email removal
    - source removal (e.g., 'Reuters', 'via')
    - punctuation and non-ASCII removal
    - whitespace normalization

    Args:
        text (str): Raw input text.

    Returns:
        str: Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = BeautifulSoup(text, "html.parser").get_text()
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b', ' ', text)
    text = re.sub(r'\b(reuters|via)\b\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def remove_stopwords(text):
    """
    Removes English stopwords from the input text string.

    Args:
        text (str): Input string.

    Returns:
        list of str: Tokens with stopwords removed.
    """
    stop_words = set(stopwords.words("english"))
    if isinstance(text, str):
        tokens = text.lower().split()
        return [word for word in tokens if word not in stop_words]
    return []


def stem_tokens(tokens):
    """
    Applies stemming to a list of tokens using the Snowball stemmer.

    Args:
        tokens (list of str): Input tokens.

    Returns:
        list of str: Stemmed tokens.
    """
    return [stemmer.stem(token) for token in tokens]


def lemmatize_tokens(tokens):
    """
    Applies lemmatization to a list of tokens using WordNet lemmatizer.

    Args:
        tokens (list of str): Input tokens.

    Returns:
        list of str: Lemmatized tokens.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]


def process_text(text: str) -> list:
    """
    Preprocesses the text for classical models (e.g., TF-IDF + Logistic Regression):
    - cleaning
    - stopword removal
    - lemmatization

    Args:
        text (str): Raw input text.

    Returns:
        list of str: List of preprocessed tokens.
    """
    cleaned = clean_text(text)
    tokens = remove_stopwords(cleaned)
    lemmatized = lemmatize_tokens(tokens)
    return lemmatized


def process_text_tokens(text: str) -> list:
    """
    Preprocesses the text by cleaning and tokenizing (no stopword removal or lemmatization).

    Args:
        text (str): Raw input text.

    Returns:
        list of str: List of cleaned tokens.
    """
    cleaned = clean_text(text)
    return cleaned.split()


def process_text_without_leaks(text: str) -> list:
    """
    Preprocesses the text for fair evaluation by removing potential leaky tokens:
    - cleaning
    - stopword removal
    - leaky word removal
    - lemmatization

    Args:
        text (str): Raw input text.

    Returns:
        list of str: List of cleaned, filtered, and lemmatized tokens.
    """
    cleaned = clean_text(text)
    tokens = remove_stopwords(cleaned)
    filtered = remove_leaky_tokens(tokens)
    lemmatized = lemmatize_tokens(filtered)
    return lemmatized
