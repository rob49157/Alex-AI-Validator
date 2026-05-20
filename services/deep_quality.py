import re
import math

_MIN_PAGES_FOR_FULL = 50
_MIN_PAGES_FOR_PARTIAL = 10


def score_content_quality(text: str, page_count: int) -> float:
    """
    Returns a quality score 0.0–1.0.
    Heuristic-only — no ML, runs in milliseconds.
    """
    words = text.split()
    if not words:
        return 0.0

    total_words = len(words)
    unique_words = len({w.lower() for w in words})

    # Text density: 1500 chars/page is a reasonable book
    chars_per_page = len(text) / max(page_count, 1)
    density_score = min(chars_per_page / 1500.0, 1.0)

    # Vocabulary richness (type-token ratio, adjusted for document length)
    ttr = unique_words / total_words
    length_factor = math.log10(max(total_words, 10)) / math.log10(10_000)
    vocab_score = min(ttr * length_factor / 0.3, 1.0)

    # Sentence length: books target ~10–20 words per sentence
    sentences = [s for s in re.split(r"[.!?]+", text) if len(s.split()) > 3]
    if sentences:
        avg_len = sum(len(s.split()) for s in sentences) / len(sentences)
    else:
        avg_len = 0.0
    sentence_score = 1.0 - min(abs(avg_len - 15) / 20.0, 1.0)

    # Noise ratio: high non-alpha, non-space ratio suggests OCR garbage or code dumps
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    noise_ratio = non_alpha / len(text)
    noise_score = max(0.0, 1.0 - noise_ratio / 0.3)

    score = (
        density_score * 0.30
        + vocab_score * 0.30
        + sentence_score * 0.15
        + noise_score * 0.25
    )
    return round(score, 4)


def estimate_completeness(text: str, page_count: int) -> str:
    """Returns 'full', 'partial', or 'excerpt'."""
    if page_count < _MIN_PAGES_FOR_PARTIAL:
        return "excerpt"

    head = text[:2_000]
    tail = text[-2_000:]

    has_start = bool(re.search(r"(?i)\b(preface|introduction|foreword|chapter\s+(1|one))\b", head))
    has_end = bool(re.search(r"(?i)\b(the end|appendix|bibliography|index|references)\b", tail))

    if page_count >= _MIN_PAGES_FOR_FULL or has_start or has_end:
        return "full"
    return "partial"
