import concurrent.futures
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["AGENT_DATA_DIR"] = tempfile.mkdtemp(prefix="agent-tests-")
from fastapi.testclient import TestClient

from app import main


def configuration(target="", action="reply"):
    return {
        "intents": [
            {"name": "order", "examples": ["где заказ", "проверь заказ"]},
            {"name": "hello", "examples": ["привет", "здравствуйте"]},
        ],
        "entities": [{"name": "number", "pattern": r"\d+"}],
        "rules": [
            {
                "state": "start",
                "intent": "order",
                "action": action,
                "text": "Готово",
                "next_state": "start",
                "target_agent": target,
            }
        ],
        "fallback": "Не понял",
        "threshold": 0.35,
    }


class AppTests(unittest.TestCase):
    def setUp(self):
        with main.db() as con:
            con.execute("DELETE FROM agents")
            con.execute("DELETE FROM sessions")
            con.execute("DELETE FROM messages")
        self.client = TestClient(main.app)

    def agent(self, name, config):
        response = self.client.post(
            "/api/agents", json={"name": name, "description": "", "config": config}
        )
        self.assertEqual(response.status_code, 201, response.text)
        a = response.json()
        model = main.DATA / (a["id"] + ".tar.gz")
        model.write_text("fake")
        with main.db() as con:
            con.execute(
                "UPDATE agents SET model_path=?, model_revision=1, model_config=?, training='ready' WHERE id=?",
                (str(model), json.dumps(config), a["id"]),
            )
        return a

    def test_validation_and_stale_model(self):
        a = self.agent("Один", configuration())
        data = {"name": "Изменён", "description": "", "config": configuration()}
        data["config"]["threshold"] = 0.7
        self.assertTrue(
            self.client.put("/api/agents/" + a["id"], json=data).json()["stale"]
        )
        data["config"]["entities"] = [{"name": "number", "pattern": "["}]
        self.assertEqual(
            self.client.put("/api/agents/" + a["id"], json=data).status_code, 422
        )
        self.assertEqual(
            self.client.get("/api/agents/" + a["id"]).json()["name"], "Изменён"
        )

    def test_handoff_context_history_and_active_agent(self):
        target = self.agent("Заказы", configuration())
        source = self.agent("Учёба", configuration(target["id"], "handoff"))
        result = {
            "intent": {"name": "order", "confidence": 0.99},
            "entities": [{"entity": "number", "value": "1234"}],
        }
        with patch.object(main, "parse", return_value=result):
            response = self.client.post(
                "/api/agents/" + source["id"] + "/chat", json={"text": "заказ 1234"}
            ).json()
            self.assertEqual(response["response"], "Готово")
            self.assertEqual(len(response["trace"]), 2)
            self.assertEqual(response["context"]["number"], "1234")
            next_response = self.client.post(
                "/api/agents/" + source["id"] + "/chat",
                json={"text": "заказ", "session_id": response["session_id"]},
            ).json()
            self.assertEqual(next_response["trace"][0]["agent"], "Заказы")
        history = self.client.get(
            "/api/sessions/" + response["session_id"] + "/messages"
        ).json()
        self.assertEqual(len(history), 2)

    def test_fallback_and_bounded_handoff(self):
        first = self.agent("Первый", configuration())
        second = self.agent("Второй", configuration(first["id"], "handoff"))
        config = configuration(second["id"], "handoff")
        with main.db() as con:
            con.execute(
                "UPDATE agents SET model_config=? WHERE id=?",
                (json.dumps(config), first["id"]),
            )
        result = {"intent": {"name": "order", "confidence": 0.9}, "entities": []}
        with patch.object(main, "parse", return_value=result):
            response = self.client.post(
                "/api/agents/" + first["id"] + "/chat", json={"text": "заказ"}
            ).json()
            self.assertIn("Превышено", response["response"])
        result["intent"]["confidence"] = 0.1
        with patch.object(main, "parse", return_value=result):
            response = self.client.post(
                "/api/agents/" + first["id"] + "/chat", json={"text": "неизвестно"}
            ).json()
            self.assertEqual(response["response"], "Не понял")

    def test_untrained_target_keeps_source_active(self):
        target = self.agent("Цель", configuration())
        source = self.agent("Источник", configuration(target["id"], "handoff"))
        with main.db() as con:
            con.execute("UPDATE agents SET model_path=NULL WHERE id=?", (target["id"],))
        with patch.object(
            main,
            "parse",
            return_value={
                "intent": {"name": "order", "confidence": 0.9},
                "entities": [],
            },
        ):
            response = self.client.post(
                "/api/agents/" + source["id"] + "/chat", json={"text": "заказ"}
            ).json()
            self.assertIn("ещё не обучен", response["response"])
            next_response = self.client.post(
                "/api/agents/" + source["id"] + "/chat",
                json={"text": "заказ", "session_id": response["session_id"]},
            ).json()
            self.assertEqual(next_response["trace"][0]["agent"], "Источник")

    def test_whitespace_and_duplicate_examples_rejected(self):
        response = self.client.post("/api/agents", json={"name": "   "})
        self.assertEqual(response.status_code, 422)
        config = configuration()
        config["intents"][1]["examples"] = ["ГДЕ ЗАКАЗ", "другая фраза"]
        self.assertEqual(
            self.client.post(
                "/api/agents", json={"name": "Агент", "config": config}
            ).status_code,
            422,
        )

    def test_rename_does_not_invalidate_model_and_revision_conflict(self):
        a = self.agent("Один", configuration())
        response = self.client.put(
            "/api/agents/" + a["id"],
            json={"name": "Два", "config": configuration(), "revision": 1},
        )
        self.assertFalse(response.json()["stale"])
        config = configuration()
        config["threshold"] = 0.7
        self.client.put(
            "/api/agents/" + a["id"],
            json={"name": "Два", "config": config, "revision": 1},
        )
        response = self.client.put(
            "/api/agents/" + a["id"],
            json={"name": "Три", "config": config, "revision": 1},
        )
        self.assertEqual(response.status_code, 409)

    def test_unknown_session_and_regex_timeout(self):
        a = self.agent("Один", configuration())
        self.assertEqual(
            self.client.post(
                "/api/agents/" + a["id"] + "/chat",
                json={"text": "заказ", "session_id": "missing"},
            ).status_code,
            404,
        )
        self.assertEqual(
            self.client.get("/api/sessions/missing/messages").status_code, 404
        )
        with self.assertRaises(main.HTTPException) as error:
            main.safe_matches("(a|aa)+$", "a" * 4000 + "!")
        self.assertEqual(error.exception.status_code, 422)

    def test_deleting_referenced_agent_is_blocked(self):
        target = self.agent("Цель", configuration())
        source = self.agent("Источник", configuration(target["id"], "handoff"))
        self.assertEqual(
            self.client.delete("/api/agents/" + target["id"]).status_code, 409
        )
        self.assertEqual(
            self.client.delete("/api/agents/" + source["id"]).status_code, 200
        )
        self.assertEqual(
            self.client.delete("/api/agents/" + target["id"]).status_code, 200
        )

    def test_concurrent_messages_keep_state(self):
        config = configuration()
        config["rules"][0]["next_state"] = "second"
        config["rules"].append(
            {**config["rules"][0], "state": "second", "next_state": "done"}
        )
        a = self.agent("Один", config)
        session_id = "concurrent-session"
        with main.db() as con:
            con.execute(
                "INSERT INTO sessions(id,agent_id,state,context) VALUES(?,?,?,?)",
                (session_id, a["id"], "start", "{}"),
            )

        def parse(*args):
            time.sleep(0.03)
            return {"intent": {"name": "order", "confidence": 0.9}, "entities": []}

        def send(_):
            return self.client.post(
                "/api/agents/" + a["id"] + "/chat",
                json={"text": "заказ", "session_id": session_id},
            ).status_code

        with (
            patch.object(main, "parse", side_effect=parse),
            concurrent.futures.ThreadPoolExecutor(2) as pool,
        ):
            self.assertEqual(list(pool.map(send, range(2))), [200, 200])
        with main.db() as con:
            self.assertEqual(
                con.execute(
                    "SELECT state FROM sessions WHERE id=?", (session_id,)
                ).fetchone()["state"],
                "done",
            )

    def test_static_files_allowlist(self):
        self.assertEqual(self.client.get("/static/app.js").status_code, 200)
        self.assertEqual(self.client.get("/static/README.md").status_code, 404)
        self.assertEqual(self.client.get("/static/.git/config").status_code, 404)

    def test_metadata_edits_have_optimistic_version(self):
        a = self.agent("Один", configuration())
        payload = {"name": "Два", "config": configuration(), "version": a["version"]}
        first = self.client.put("/api/agents/" + a["id"], json=payload)
        self.assertFalse(first.json()["stale"])
        self.assertEqual(
            self.client.put("/api/agents/" + a["id"], json=payload).status_code, 409
        )

    def test_schema_inspection_does_not_interrupt_training(self):
        a = self.agent("Один", configuration())
        with main.db() as con:
            con.execute("UPDATE agents SET training='running' WHERE id=?", (a["id"],))
        main.init_db()
        self.assertEqual(main.get_agent(a["id"])["training"], "running")
        main.init_db(recover=True)
        self.assertEqual(main.get_agent(a["id"])["training"], "failed")

    def test_hosted_api_requires_secret(self):
        with patch.dict(os.environ, {"ATLAS_API_TOKEN": "test-only-token"}):
            self.assertEqual(self.client.get("/api/agents").status_code, 401)
            self.assertEqual(
                self.client.get(
                    "/api/agents", headers={"Authorization": "Bearer wrong"}
                ).status_code,
                401,
            )
            self.assertEqual(
                self.client.get(
                    "/api/agents", headers={"Authorization": "Bearer test-only-token"}
                ).status_code,
                200,
            )

    def test_training_failure_keeps_model(self):
        a = self.agent("Первый", configuration())
        with patch.object(
            main.subprocess, "run", side_effect=FileNotFoundError("Rasa отсутствует")
        ):
            main.train_worker(a["id"], 2, configuration())
        saved = main.get_agent(a["id"])
        self.assertEqual(saved["training"], "failed")
        self.assertEqual(saved["model_revision"], 1)
        self.assertTrue(Path(saved["model_path"]).is_file())


if __name__ == "__main__":
    unittest.main()
