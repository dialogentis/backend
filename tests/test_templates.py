from uuid import uuid4

import pytest

from dialogentis.agents import empty_config
from dialogentis.flow_validation import validate_flow
from dialogentis.slots import Slot
from dialogentis.templates import TemplateError, parse_template, render_template


def slot(code, **values):
    return Slot.model_validate({"id": str(uuid4()), "code": code, **values})


def test_plain_text_rendering_defaults_optional_values_and_unicode():
    slots = [
        slot("city", required=True),
        slot("amount", value_type="number", default="12.50"),
        slot("consent", value_type="boolean"),
        slot("comment"),
    ]
    template = "Доставка 👩🏽‍💻 в {{ city }}, сумма {{amount}}. Согласие: {{consent}}. {{comment}}"
    assert (
        render_template(template, slots, {"city": "Москва", "consent": False})
        == "Доставка 👩🏽‍💻 в Москва, сумма 12.50. Согласие: нет. "
    )
    assert render_template("{{city}} / {{city}}", slots, {"city": "Омск"}) == "Омск / Омск"
    # A value is returned as plain text; it is not interpreted as a template or HTML.
    assert render_template("{{city}}", slots, {"city": "<b>{{amount}}</b>"}) == "<b>{{amount}}</b>"


@pytest.mark.parametrize(
    "text",
    [
        "{{city",
        "city}}",
        "{{city.name}}",
        "{{city()}}",
        "{{{city}}}",
        "{{unknown}}",
        "<b>text</b>",
        "<!-- hidden -->",
        "<!doctype html>",
    ],
)
def test_invalid_templates_are_rejected(text):
    with pytest.raises(TemplateError):
        parse_template(text, [slot("city")])


def test_rendering_requires_valid_values_and_does_not_expose_them_in_errors():
    slots = [slot("count", value_type="integer", required=True, sensitive=True)]
    for values, code in [
        ({}, "required_slot_missing"),
        ({"count": "SECRET_SENTINEL"}, "invalid_slot_value"),
        ({"count": True}, "invalid_slot_value"),
    ]:
        with pytest.raises(TemplateError) as error:
            render_template("{{count}}", slots, values)
        assert error.value.code == code and "SECRET_SENTINEL" not in str(error.value)
        assert error.value.__context__ is None or error.value.__suppress_context__
    assert render_template("{{count}}", slots, {"count": 0}) == "0"


def test_templates_have_bounded_input_and_output_and_unique_slots():
    city = slot("city")
    with pytest.raises(TemplateError, match="template_too_long"):
        parse_template("x" * 4001, [city])
    with pytest.raises(TemplateError, match="rendered_text_too_long"):
        render_template("{{city}}{{city}}{{city}}", [city], {"city": "x" * 2000})
    with pytest.raises(TemplateError, match="invalid_slot_definitions"):
        parse_template("text", [city, city])


def test_flow_validation_reports_text_and_template_errors():
    config = empty_config()
    config["slots"] = [{"id": str(uuid4()), "code": "city"}]
    fallback = config["flow"]["nodes"][2]
    fallback["config"]["text"] = "Город: {{city}}"
    assert not validate_flow(config)[0]
    fallback["config"]["text"] = "Город: {{other}}"
    assert "unknown_template_slot" in {issue.code for issue in validate_flow(config)[0]}
    fallback["config"]["text"] = "<b>Ошибка</b>"
    assert "html_not_allowed" in {issue.code for issue in validate_flow(config)[0]}
    fallback["config"]["text"] = " "
    assert "missing_text" in {issue.code for issue in validate_flow(config)[0]}
