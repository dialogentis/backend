from unicodedata import category


def _word_character(character: str) -> bool:
    return category(character)[0] in {"L", "N", "M"} or category(character) == "Pc"


def _regional_indicator(character: str) -> bool:
    return "\U0001f1e6" <= character <= "\U0001f1ff"


def token_spans(text: str) -> list[tuple[int, int]]:
    """Return half-open Unicode code-point offsets for the shared CRF tokenizer."""
    # shortcut: simplified grapheme grouping; upgrade if additional scripts need full UAX #29.
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(text):
        if text[index].isspace():
            index += 1
            continue
        start = index
        index += 1
        if _word_character(text[start]):
            while index < len(text) and _word_character(text[index]):
                index += 1
        else:
            if (
                _regional_indicator(text[start])
                and index < len(text)
                and _regional_indicator(text[index])
            ):
                index += 1
            while index < len(text):
                character = text[index]
                if category(character)[0] == "M" or "\U0001f3fb" <= character <= "\U0001f3ff":
                    index += 1
                elif (
                    character == "\u200d"
                    and index + 1 < len(text)
                    and not text[index + 1].isspace()
                ):
                    index += 2
                else:
                    break
        spans.append((start, index))
    return spans
