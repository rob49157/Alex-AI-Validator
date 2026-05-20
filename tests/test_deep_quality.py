import pytest
from services.deep_quality import score_content_quality, estimate_completeness


# ── score_content_quality ──────────────────────────────────────────────────────

BOOK_PARAGRAPH = (
    "The expedition set off at dawn, navigating through dense jungle terrain. "
    "Rivers carved deep gorges into the limestone plateau, making progress slow. "
    "Each evening the team catalogued specimens and updated their field notes carefully. "
)


def test_empty_text_scores_zero():
    assert score_content_quality("", 10) == 0.0


def test_real_book_text_scores_well():
    text = BOOK_PARAGRAPH * 100
    score = score_content_quality(text, 20)
    assert score >= 0.5


def test_gibberish_scores_low():
    gibberish = "##@!$$%^^^&**()___++==[]{};':\",./<>?~` " * 200
    score = score_content_quality(gibberish, 5)
    assert score < 0.4


def test_score_is_bounded():
    text = BOOK_PARAGRAPH * 200
    score = score_content_quality(text, 50)
    assert 0.0 <= score <= 1.0


def test_single_page_low_text_penalised():
    sparse = "word " * 10
    score = score_content_quality(sparse, 50)
    assert score < 0.5


# ── estimate_completeness ──────────────────────────────────────────────────────

def test_short_doc_is_excerpt():
    result = estimate_completeness(BOOK_PARAGRAPH * 5, page_count=3)
    assert result == "excerpt"


def test_long_doc_is_full():
    text = BOOK_PARAGRAPH * 300
    result = estimate_completeness(text, page_count=200)
    assert result == "full"


def test_doc_with_intro_marker_is_full():
    text = "Introduction\n" + BOOK_PARAGRAPH * 30
    result = estimate_completeness(text, page_count=15)
    assert result == "full"


def test_doc_with_bibliography_end_is_full():
    text = BOOK_PARAGRAPH * 30 + "\nReferences\n..."
    result = estimate_completeness(text, page_count=15)
    assert result == "full"


def test_medium_doc_without_markers_is_partial():
    text = BOOK_PARAGRAPH * 10  # no intro/end markers
    result = estimate_completeness(text, page_count=25)
    assert result == "partial"
