"""Build portable agent files with examples and test phrases."""

import argparse
import json
import re
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from dialogentis.agents import empty_config
from dialogentis.datasets.export_worker import serialize_export
from dialogentis.datasets.schema import Dataset, validate_dataset

FIXTURES = Path(__file__).with_name("fixtures.json")


def identifier(*parts: str):
    return uuid5(NAMESPACE_URL, "dialogentis/benchmark/" + "/".join(parts))


def build_dataset(name: str, fixture: dict) -> Dataset:
    entity = fixture["entity"]
    entity_id = identifier(name, "entity", entity["code"])
    intents = [
        {"id": identifier(name, "intent", item["code"]), "code": item["code"], "name": item["name"]}
        for item in fixture["intents"]
    ]
    examples = []
    for item in fixture["intents"]:
        for split in ("train", "test"):
            for text in item[split]:
                spans = []
                for term in entity["values"]:
                    for match in re.finditer(r"(?<!\w)" + re.escape(term) + r"(?!\w)", text, re.I):
                        spans.append(
                            {"entity_id": entity_id, "start": match.start(), "end": match.end()}
                        )
                examples.append(
                    {
                        "id": identifier(name, split, text),
                        "text": text,
                        "intent_id": identifier(name, "intent", item["code"]),
                        "split": split,
                        "spans": spans,
                    }
                )
    for text in fixture["unknown"]:
        examples.append(
            {
                "id": identifier(name, "unknown", text),
                "text": text,
                "expected_fallback": True,
                "split": "test",
                "spans": [],
            }
        )
    return Dataset.model_validate(
        {
            "intents": intents,
            "entities": [
                {
                    "id": entity_id,
                    "code": entity["code"],
                    "name": entity["name"],
                    "method": "dictionary",
                    "value_type": "string",
                    "dictionary": [{"value": term} for term in entity["values"]],
                }
            ],
            "examples": examples,
        }
    )


DEVELOPMENT = FIXTURES.with_name("demo_training.json")
ANSWERS = {
    "order_status": "Статус заказа доступен в личном кабинете в разделе «Заказы».",
    "delivery": "Срок доставки указан в заказе. Для уточнения адреса откройте карточку доставки.",
    "return_order": "Откройте полученный заказ и выберите «Оформить возврат».",
    "payment": "Доступные способы оплаты появятся на странице оформления заказа.",
    "cancel_order": "Если заказ ещё можно отменить, кнопка появится в карточке заказа.",
    "enrollment": "Откройте страницу выбранного курса и заполните заявку на участие.",
    "schedule": "Расписание и ближайшее занятие доступны в календаре учебной группы.",
    "course_price": "Актуальная стоимость и условия оплаты указаны на странице курса.",
    "certificate": "После завершения программы проверьте раздел документов в учебном кабинете.",
    "format": "Формат и требования к подключению указаны в описании выбранного курса.",
}


def load_demo_fixtures(*, training_only: bool = False) -> dict:
    frozen = json.loads(FIXTURES.read_text())
    development = json.loads(DEVELOPMENT.read_text())
    return {
        name: {
            **fixture,
            "unknown": [] if training_only else fixture["unknown"],
            "intents": [
                {
                    **intent,
                    "train": intent["train"] + development[name]["train"][intent["code"]],
                    "test": [] if training_only else intent["test"],
                }
                for intent in fixture["intents"]
            ],
        }
        for name, fixture in frozen.items()
    }


def demo_config(name: str, fixture: dict) -> dict:
    dataset = build_dataset(name, fixture)
    errors, _ = validate_dataset(dataset)
    if errors:
        raise ValueError(f"{name}: invalid dataset: {[(item.path, item.code) for item in errors]}")
    config = empty_config()
    config.update(dataset.model_dump(mode="json"))
    flow = config["flow"]
    switch, end = flow["nodes"][1], flow["nodes"][3]
    end["position"] = {"x": 750, "y": 0}
    for row, intent in enumerate(dataset.intents, start=1):
        answer_id = str(identifier(name, "answer", intent.code))
        flow["nodes"].append(
            {
                "id": answer_id,
                "type": "say",
                "position": {"x": 500, "y": 110 * row},
                "config": {"text": ANSWERS[intent.code]},
            }
        )
        flow["edges"].extend(
            [
                {
                    "id": str(identifier(name, "branch", intent.code)),
                    "source": switch["id"],
                    "source_handle": str(intent.id),
                    "target": answer_id,
                },
                {
                    "id": str(identifier(name, "finish", intent.code)),
                    "source": answer_id,
                    "source_handle": "next",
                    "target": end["id"],
                },
            ]
        )
    return config


def main() -> None:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("agent", choices=("order_support", "education"))
    name = command.parse_args().agent
    print(serialize_export(demo_config(name, load_demo_fixtures()[name]), "json").decode())


if __name__ == "__main__":
    main()
