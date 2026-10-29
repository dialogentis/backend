"""Добавить демонстрационных агентов в пустую базу без запуска сервера."""

import json
import uuid

from app.main import db


def add(con, agent_id, name, description, config):
    con.execute(
        "INSERT INTO agents(id,name,description,config) VALUES(?,?,?,?)",
        (agent_id, name, description, json.dumps(config, ensure_ascii=False)),
    )


with db() as con:
    if con.execute("SELECT COUNT(*) FROM agents").fetchone()[0]:
        raise SystemExit("База не пуста; демонстрационные данные не добавлены.")
    study, orders = str(uuid.uuid4()), str(uuid.uuid4())
    common = {
        "fallback": "Не удалось понять вопрос. Попробуйте ещё раз.",
        "threshold": 0.35,
        "entities": [],
    }
    add(
        con,
        study,
        "Учебный помощник",
        "Отвечает на вопросы о расписании и передаёт запросы о заказах.",
        {
            **common,
            "intents": [
                {
                    "name": "schedule",
                    "examples": [
                        "покажи расписание",
                        "когда занятия",
                        "расписание занятий",
                        "во сколько урок",
                        "когда начинается пара",
                    ],
                },
                {
                    "name": "order",
                    "examples": [
                        "где мой заказ",
                        "статус заказа",
                        "как получить заказ",
                        "проверь заказ",
                        "узнать о доставке",
                    ],
                },
            ],
            "rules": [
                {
                    "state": "start",
                    "intent": "schedule",
                    "action": "reply",
                    "text": "Расписание доступно в учебном кабинете.",
                    "next_state": "start",
                    "target_agent": "",
                },
                {
                    "state": "start",
                    "intent": "order",
                    "action": "handoff",
                    "text": "",
                    "next_state": "start",
                    "target_agent": orders,
                },
            ],
        },
    )
    add(
        con,
        orders,
        "Помощник по заказам",
        "Распознаёт номер заказа и помогает с вопросами о доставке.",
        {
            **common,
            "entities": [{"name": "order_number", "pattern": r"\b\d{4,8}\b"}],
            "intents": [
                {
                    "name": "order",
                    "examples": [
                        "где мой заказ",
                        "статус заказа 1234",
                        "как получить заказ",
                        "проверь заказ 5678",
                        "узнать о доставке",
                    ],
                },
                {
                    "name": "greeting",
                    "examples": [
                        "привет",
                        "здравствуйте",
                        "добрый день",
                        "доброе утро",
                        "приветствую",
                    ],
                },
            ],
            "rules": [
                {
                    "state": "start",
                    "intent": "order",
                    "action": "reply",
                    "text": "Проверьте статус заказа в личном кабинете.",
                    "next_state": "start",
                    "target_agent": "",
                },
                {
                    "state": "start",
                    "intent": "greeting",
                    "action": "reply",
                    "text": "Здравствуйте! Помогу с вопросами о заказах.",
                    "next_state": "start",
                    "target_agent": "",
                },
            ],
        },
    )
print("Созданы два демонстрационных агента. Обучите каждого в интерфейсе.")
