from langdetect import detect, LangDetectException


def detect_language(text: str) -> str | None:
    """Returns ISO 639-1 language code, or None if detection fails."""
    try:
        return detect(text[:5_000])
    except LangDetectException:
        return None
