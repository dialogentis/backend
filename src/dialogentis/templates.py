"""Plain-text templates with explicit slot references and bounded rendering."""

import re
from html.parser import HTMLParser
from typing import Any

from pydantic import ValidationError

from dialogentis.slots import Slot

MAX_MESSAGE_LENGTH = 4000
SLOT_CODE = re.compile(r"[a-z][a-z0-9_]{1,63}\Z")


class TemplateError(ValueError):
    def __init__(self, code: str):
        self.code = code
        super().__init__(code)


class _PlainText(HTMLParser):
    def handle_starttag(self, tag, attrs):
        raise TemplateError("html_not_allowed")

    def handle_endtag(self, tag):
        raise TemplateError("html_not_allowed")

    def handle_startendtag(self, tag, attrs):
        raise TemplateError("html_not_allowed")

    def handle_comment(self, data):
        raise TemplateError("html_not_allowed")

    def handle_decl(self, decl):
        raise TemplateError("html_not_allowed")

    def handle_pi(self, data):
        raise TemplateError("html_not_allowed")


def parse_template(text: str, slots: list[Slot]) -> list[tuple[str, bool]]:
    if len(text) > MAX_MESSAGE_LENGTH:
        raise TemplateError("template_too_long")
    try:
        text.encode("utf-8")
    except UnicodeError:
        raise TemplateError("invalid_template_text") from None
    parser = _PlainText(convert_charrefs=False)
    parser.feed(text)
    parser.close()
    codes = {slot.code for slot in slots}
    if len(codes) != len(slots) or len({slot.id for slot in slots}) != len(slots):
        raise TemplateError("invalid_slot_definitions")
    tokens: list[tuple[str, bool]] = []
    cursor = 0
    while cursor < len(text):
        start, unexpected_end = text.find("{{", cursor), text.find("}}", cursor)
        if unexpected_end >= 0 and (start < 0 or unexpected_end < start):
            raise TemplateError("invalid_template")
        if start < 0:
            tokens.append((text[cursor:], False))
            break
        tokens.append((text[cursor:start], False))
        end = text.find("}}", start + 2)
        if end < 0:
            raise TemplateError("invalid_template")
        code = text[start + 2 : end].strip()
        if not SLOT_CODE.fullmatch(code):
            raise TemplateError("invalid_template")
        if code not in codes:
            raise TemplateError("unknown_template_slot")
        tokens.append((code, True))
        cursor = end + 2
    return tokens


def render_template(text: str, slots: list[Slot], values: dict[str, Any]) -> str:
    tokens = parse_template(text, slots)
    by_code = {slot.code: slot for slot in slots}
    result: list[str] = []
    size = 0
    for token, placeholder in tokens:
        if placeholder:
            slot = by_code[token]
            value = values.get(token, slot.default)
            if value is None:
                if slot.required:
                    raise TemplateError("required_slot_missing")
                segment = ""
            else:
                try:
                    Slot.model_validate({**slot.model_dump(), "default": value})
                except ValidationError:
                    raise TemplateError("invalid_slot_value") from None
                segment = ("да" if value else "нет") if type(value) is bool else str(value)
        else:
            segment = token
        size += len(segment)
        if size > MAX_MESSAGE_LENGTH:
            raise TemplateError("rendered_text_too_long")
        result.append(segment)
    return "".join(result)
