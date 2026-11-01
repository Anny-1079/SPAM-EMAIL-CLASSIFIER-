"""
Text cleaning and preprocessing shared by training (spam_classifier.py)
and prediction (predict.py), so both always use identical steps.

Pipeline for one email:
    subject + body
      -> truncate very long bodies
      -> lowercase
      -> strip HTML tags
      -> replace URLs and email addresses with placeholder tokens
      -> tokenize (letters only; digits and punctuation dropped)
      -> remove stopwords and very short / very long tokens
      -> Porter stemming  ("winning", "wins" -> "win")
"""

from __future__ import annotations

import re
from functools import lru_cache

import pandas as pd
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

MAX_CHARS = 10_000  # a few emails are 200k+ characters of forwarded text; the start is enough
MIN_TOKEN_LEN = 2
MAX_TOKEN_LEN = 25  # longer "words" are encoded garbage, not language

HTML_RE = re.compile(r"<[^>]+>")
URL_RE = re.compile(r"(https?://\S+|www\.\S+)")
EMAIL_RE = re.compile(r"\S+@\S+")
TOKEN_RE = re.compile(r"[a-z]+")

STOPWORDS = frozenset(ENGLISH_STOP_WORDS)
_stemmer = PorterStemmer()


@lru_cache(maxsize=None)
def _stem(token: str) -> str:
    """Stemming is the slowest step, so each distinct word is stemmed only once."""
    return _stemmer.stem(token)


def tokenize(text: str) -> list[str]:
    """Clean raw email text and return the list of processed tokens."""
    text = text[:MAX_CHARS].lower()
    text = HTML_RE.sub(" ", text)
    text = URL_RE.sub(" urltoken ", text)
    text = EMAIL_RE.sub(" emailtoken ", text)
    tokens = TOKEN_RE.findall(text)
    return [
        _stem(t)
        for t in tokens
        if t not in STOPWORDS and MIN_TOKEN_LEN <= len(t) <= MAX_TOKEN_LEN
    ]


def combine(subject: object, message: object) -> str:
    """Join subject and body into one string (either may be missing)."""
    parts = [x for x in (subject, message) if isinstance(x, str) and x.strip()]
    return " . ".join(parts)


def preprocess(subject: object, message: object) -> str:
    """Full preprocessing of one email -> space-separated processed tokens."""
    return " ".join(tokenize(combine(subject, message)))


def preprocess_frame(df: pd.DataFrame, subject_col: str = "Subject", message_col: str = "Message") -> pd.Series:
    """Vectorised helper: preprocess every row of a DataFrame."""
    return pd.Series(
        [preprocess(s, m) for s, m in zip(df[subject_col], df[message_col])],
        index=df.index,
        name="clean_text",
    )


if __name__ == "__main__":
    demo = "Subject: WINNING!!! Click http://spam.example/win now to claim $1,000,000 <b>FREE</b>"
    print("raw  :", demo)
    print("clean:", preprocess(None, demo))
