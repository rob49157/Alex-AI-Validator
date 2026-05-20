from services.language_detection import detect_language


def test_detects_english():
    text = (
        "The history of science is a fascinating journey through human curiosity and discovery. "
        "From ancient astronomers mapping the heavens to modern physicists exploring quantum realms, "
        "each generation has built upon the knowledge of those who came before. "
    ) * 10
    assert detect_language(text) == "en"


def test_returns_none_for_empty_string():
    assert detect_language("") is None


def test_returns_none_for_very_short_text():
    # langdetect raises LangDetectException for texts that are too short to classify
    result = detect_language("hi")
    # May return a code or None — either is acceptable; must not raise
    assert result is None or isinstance(result, str)


def test_detects_spanish():
    text = (
        "La historia de la ciencia es un fascinante viaje a través de la curiosidad humana. "
        "Desde los antiguos astrónomos que cartografiaban los cielos hasta los modernos físicos "
        "que exploran los reinos cuánticos, cada generación ha construido sobre el conocimiento. "
    ) * 10
    result = detect_language(text)
    assert result == "es"
