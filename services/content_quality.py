import re
import math
from dataclasses import dataclass


@dataclass
class QualityResult:
    has_readable_text: bool
    text_density_ok: bool
    flags: list[str]


_MIN_TOTAL_CHARS = 200
_GIBBERISH_RATIO_THRESHOLD = 0.4  # >40% non-alphabetic chars = suspicious


def check_basic_quality(text: str, page_count: int) -> QualityResult:
    """
    Fast, heuristic-only content checks for Tier 1.
    No ML — runs in milliseconds.
    """
    flags: list[str] = []

    if not text or len(text.strip()) < _MIN_TOTAL_CHARS:
        return QualityResult(has_readable_text=False, text_density_ok=False, flags=["no_readable_text"])

    # Check for gibberish: ratio of non-alphabetic, non-space characters
    non_alpha = sum(1 for c in text if not c.isalpha() and not c.isspace())
    gibberish_ratio = non_alpha / len(text)
    if gibberish_ratio > _GIBBERISH_RATIO_THRESHOLD:
        flags.append("high_non_alpha_ratio")

    # Check text density relative to page count
    chars_per_page = len(text) / max(page_count, 1)
    text_density_ok = chars_per_page >= 100
    if not text_density_ok:
        flags.append("low_text_density")

    return QualityResult(
        has_readable_text=True,
        text_density_ok=text_density_ok,
        flags=flags,
    )
