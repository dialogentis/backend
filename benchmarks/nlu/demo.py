"""Build the portable agent files and evaluate their frozen test sets."""

import argparse
import json

from benchmarks.nlu.evaluate import FIXTURES, build_dataset, identifier
from dialogentis.agents import empty_config
from dialogentis.datasets.export_worker import serialize_export
from dialogentis.datasets.schema import validate_dataset
from dialogentis.nlu.adapter import NLUAdapter
from dialogentis.nlu.intent import Thresholds
from dialogentis.nlu.settings import NLUSettings

DEVELOPMENT = FIXTURES.with_name("demo_training.json")
SETTINGS = {
    "order_support": NLUSettings(score=0.30, margin=0.05, similarity=0.20),
    "education": NLUSettings(
        score=0.50, margin=0.16, similarity=0.15, vocabulary_coverage=0.55, classifier_c=10
    ),
}
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
    config["nlu_settings"] = SETTINGS[name].model_dump()
    config["quality_settings"] = {"intent_f1": 0.85, "entity_f1": 0.85}
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
    command.add_argument(
        "agent", choices=("order_support", "education", "evaluate"), help="Agent JSON or evaluation"
    )
    args = command.parse_args()
    fixtures = load_demo_fixtures()
    if args.agent != "evaluate":
        print(serialize_export(demo_config(args.agent, fixtures[args.agent]), "json").decode())
        return
    results = {}
    for name, fixture in fixtures.items():
        dataset = build_dataset(name, fixture)
        settings = SETTINGS[name]
        model = NLUAdapter.train(
            identifier(name, "demo"),
            dataset,
            Thresholds.from_settings(settings),
            classifier_c=settings.classifier_c,
        )
        report = model.evaluate(dataset)
        results[name] = {
            "settings": settings.model_dump(),
            "training_examples": model.intent_model.manifest["training_examples"],
            "intent_macro_f1": report.intents.macro_f1,
            "entity_f1": report.entities.f1,
            "known_test_count": report.intents.known_count,
            "unknown_test_count": report.intents.unknown_count,
            "unknown_rejection": report.intents.unknown_rejection,
        }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
