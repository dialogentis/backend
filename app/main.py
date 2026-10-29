import hmac
import json
import os
import re
import sqlite3
import subprocess
import threading
import uuid
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import Literal

import regex
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from app import inference

ROOT = Path(__file__).resolve().parents[1]
FRONTEND = Path(os.environ.get("FRONTEND_DIR", ROOT.parent / "frontend"))
DATA = Path(os.environ.get("AGENT_DATA_DIR", ROOT / "data")).resolve()
DATA.mkdir(parents=True, exist_ok=True)
DB = DATA / "agents.sqlite3"
RASA_PYTHON = os.environ.get(
    "RASA_PYTHON",
    str(
        ROOT
        / "rasa"
        / ".venv"
        / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    ),
)


@asynccontextmanager
async def lifespan(app):
    init_db(recover=True)
    yield
    inference.close_all()


app = FastAPI(title="dialogentis API", lifespan=lifespan)


@app.middleware("http")
async def authenticate_api(request, call_next):
    token = os.environ.get("ATLAS_API_TOKEN")
    if token and request.url.path.startswith("/api/"):
        supplied = request.headers.get("Authorization", "")
        if not hmac.compare_digest(supplied.encode(), ("Bearer " + token).encode()):
            return JSONResponse({"detail": "Требуется авторизация"}, status_code=401)
    return await call_next(request)


lock = threading.RLock()
session_locks = [threading.Lock() for _ in range(64)]


@contextmanager
def db():
    connection = sqlite3.connect(DB, timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def init_db(recover=False):
    with db() as con:
        con.executescript("""CREATE TABLE IF NOT EXISTS agents (id TEXT PRIMARY KEY, name TEXT NOT NULL, description TEXT NOT NULL, config TEXT NOT NULL, revision INTEGER NOT NULL DEFAULT 1, model_revision INTEGER, model_path TEXT, model_config TEXT, training TEXT NOT NULL DEFAULT 'idle', error TEXT NOT NULL DEFAULT '');
        CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY, agent_id TEXT NOT NULL, state TEXT NOT NULL, context TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS messages (id INTEGER PRIMARY KEY AUTOINCREMENT, session_id TEXT NOT NULL, text TEXT NOT NULL, response TEXT NOT NULL, trace TEXT NOT NULL, created_at TEXT DEFAULT CURRENT_TIMESTAMP);""")
        columns = {row["name"] for row in con.execute("PRAGMA table_info(sessions)")}
        if "active_agent" not in columns:
            con.execute("ALTER TABLE sessions ADD COLUMN active_agent TEXT")
        agent_columns = {
            row["name"] for row in con.execute("PRAGMA table_info(agents)")
        }
        if "version" not in agent_columns:
            con.execute(
                "ALTER TABLE agents ADD COLUMN version INTEGER NOT NULL DEFAULT 1"
            )
        if recover:
            con.execute(
                "UPDATE agents SET training='failed', error='Обучение прервано перезапуском сервера' WHERE training='running'"
            )


class Intent(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    examples: list[str] = Field(min_length=2, max_length=1000)


class Entity(BaseModel):
    name: str = Field(pattern=r"^[a-zA-Z][a-zA-Z0-9_]*$")
    pattern: str = Field(min_length=1, max_length=1000)


class Rule(BaseModel):
    state: str = Field(default="start", min_length=1)
    intent: str
    action: Literal["reply", "ask", "handoff"]
    text: str = ""
    next_state: str = "start"
    target_agent: str = ""

    @field_validator("state", "next_state", "intent", "target_agent")
    @classmethod
    def trim_identifiers(cls, value):
        return value.strip()


class Config(BaseModel):
    intents: list[Intent] = Field(default_factory=list)
    entities: list[Entity] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    fallback: str = "Извините, я не понял запрос. Попробуйте сформулировать его иначе."
    threshold: float = Field(default=0.35, ge=0, le=1)

    @model_validator(mode="after")
    def validate_config(self):
        for group in (self.intents, self.entities):
            names = [item.name for item in group]
            if len(names) != len(set(names)):
                raise ValueError(
                    "Названия намерений и сущностей должны быть уникальными"
                )
        phrases = set()
        if not self.fallback.strip():
            raise ValueError("Укажите ответ по умолчанию")
        for intent in self.intents:
            if any(not e.strip() or "\n" in e for e in intent.examples):
                raise ValueError("Примеры должны быть непустыми однострочными фразами")
            for example in intent.examples:
                normalized = example.strip().casefold()
                if normalized in phrases:
                    raise ValueError(
                        "Примеры фраз не должны повторяться, в том числе между намерениями"
                    )
                phrases.add(normalized)
                if len(example) > 4000 or "\r" in example:
                    raise ValueError(
                        "Пример должен быть одной строкой длиной до 4000 символов"
                    )
        for ent in self.entities:
            try:
                re.compile(ent.pattern)
                regex.compile(ent.pattern)
            except re.error as e:
                raise ValueError(f"Неверное регулярное выражение {ent.name}: {e}")
        seen = set()
        for rule in self.rules:
            if not rule.state.strip() or not rule.next_state.strip():
                raise ValueError("Состояния не могут быть пустыми")
            if rule.intent not in {i.name for i in self.intents}:
                raise ValueError(f"Неизвестное намерение: {rule.intent}")
            if (rule.state, rule.intent) in seen:
                raise ValueError("Для состояния и намерения допускается одно правило")
            seen.add((rule.state, rule.intent))
            if rule.action == "handoff" and not rule.target_agent:
                raise ValueError("Укажите целевого агента для передачи")
            if rule.action != "handoff" and not rule.text.strip():
                raise ValueError("Укажите текст ответа")
        return self


class AgentInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    description: str = ""
    config: Config = Field(default_factory=Config)
    revision: int | None = None
    version: int | None = None

    @field_validator("name")
    @classmethod
    def nonblank_name(cls, value):
        if not value.strip():
            raise ValueError("Название не может состоять из пробелов")
        return value.strip()


class ChatInput(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    session_id: str | None = None


def get_agent(agent_id):
    with db() as con:
        row = con.execute("SELECT * FROM agents WHERE id=?", (agent_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Агент не найден")
    return dict(row)


def public(row):
    obj = dict(row)
    obj["config"] = json.loads(obj["config"])
    obj["stale"] = obj["model_revision"] != obj["revision"]
    obj["has_model"] = bool(obj["model_path"] and Path(obj["model_path"]).is_file())
    obj.pop("model_path", None)
    obj.pop("model_config", None)
    return obj


def rasa_files(config, path):
    path.mkdir(parents=True, exist_ok=True)
    nlu = [
        {"intent": i["name"], "examples": "\n".join("- " + e for e in i["examples"])}
        for i in config["intents"]
    ]
    nlu += [
        {"regex": e["name"], "examples": "- " + e["pattern"]}
        for e in config["entities"]
    ]
    pipeline = [
        {"name": "WhitespaceTokenizer"},
        {"name": "RegexFeaturizer"},
        {
            "name": "CountVectorsFeaturizer",
            "analyzer": "char_wb",
            "min_ngram": 1,
            "max_ngram": 4,
        },
        {"name": "DIETClassifier", "epochs": 30, "entity_recognition": False},
    ]
    for name, content in [
        ("nlu.yml", {"version": "3.1", "nlu": nlu}),
        ("config.yml", {"language": "ru", "pipeline": pipeline, "policies": []}),
    ]:
        (path / name).write_text(
            yaml.safe_dump(content, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )


def train_worker(agent_id, revision, config):
    work = DATA / "models" / agent_id / str(uuid.uuid4())
    try:
        rasa_files(config, work)
        command = [
            RASA_PYTHON,
            "-m",
            "rasa",
            "train",
            "nlu",
            "--config",
            str(work / "config.yml"),
            "--nlu",
            str(work / "nlu.yml"),
            "--out",
            str(work / "out"),
            "--fixed-model-name",
            "model",
        ]
        result = subprocess.run(
            command,
            capture_output=True,
            check=False,
            text=True,
            timeout=1800,
            cwd=work,
            env={
                **os.environ,
                "RASA_TELEMETRY_ENABLED": "false",
                "TF_CPP_MIN_LOG_LEVEL": "2",
                "OMP_NUM_THREADS": "1",
                "TF_NUM_INTRAOP_THREADS": "1",
                "TF_NUM_INTEROP_THREADS": "1",
            },
        )
        model = work / "out" / "model.tar.gz"
        if result.returncode or not model.is_file():
            raise RuntimeError(
                (result.stderr + "\n" + result.stdout)[-5000:]
                or "Rasa не создала модель"
            )
        with db() as con:
            con.execute(
                "UPDATE agents SET model_path=?, model_revision=?, model_config=?, training='ready', error='' WHERE id=?",
                (
                    str(model),
                    revision,
                    json.dumps(config, ensure_ascii=False),
                    agent_id,
                ),
            )
    except Exception as exc:  # noqa: BLE001 - Persist unexpected background-job errors.
        with db() as con:
            con.execute(
                "UPDATE agents SET training='failed', error=? WHERE id=?",
                (str(exc)[-5000:], agent_id),
            )


def parse(model, text):
    try:
        return inference.parse(RASA_PYTHON, ROOT / "app" / "rasa_parse.py", model, text)
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(503, f"Ошибка анализа Rasa: {exc}") from exc


@app.get("/api/agents")
def list_agents():
    with db() as con:
        return [
            public(row) for row in con.execute("SELECT * FROM agents ORDER BY name")
        ]


@app.post("/api/agents", status_code=201)
def create_agent(data: AgentInput):
    agent_id = str(uuid.uuid4())
    with db() as con:
        con.execute(
            "INSERT INTO agents(id,name,description,config) VALUES(?,?,?,?)",
            (
                agent_id,
                data.name.strip(),
                data.description,
                data.config.model_dump_json(),
            ),
        )
    return public(get_agent(agent_id))


@app.get("/api/agents/{agent_id}")
def read_agent(agent_id: str):
    return public(get_agent(agent_id))


@app.put("/api/agents/{agent_id}")
def update_agent(agent_id: str, data: AgentInput):
    row = get_agent(agent_id)
    with db() as con:
        if (data.version is not None and data.version != row["version"]) or (
            data.revision is not None and data.revision != row["revision"]
        ):
            raise HTTPException(
                409, "Настройки изменены в другом окне. Откройте агента повторно."
            )
        changed = json.loads(row["config"]) != data.config.model_dump()
        result = con.execute(
            "UPDATE agents SET name=?,description=?,config=?,revision=revision+?,version=version+1 WHERE id=? AND version=?",
            (
                data.name.strip(),
                data.description,
                data.config.model_dump_json(),
                int(changed),
                agent_id,
                row["version"],
            ),
        )
        if result.rowcount != 1:
            raise HTTPException(409, "Настройки изменились. Откройте агента повторно.")
    return public(get_agent(agent_id))


@app.delete("/api/agents/{agent_id}")
def delete_agent(agent_id: str):
    with lock:
        return remove_agent(agent_id)


def remove_agent(agent_id: str):
    row = get_agent(agent_id)
    if row["training"] == "running":
        raise HTTPException(409, "Дождитесь завершения обучения")
    with db() as con:
        for source in con.execute(
            "SELECT id, config FROM agents WHERE id != ?", (agent_id,)
        ):
            if any(
                r["action"] == "handoff" and r["target_agent"] == agent_id
                for r in json.loads(source["config"])["rules"]
            ):
                raise HTTPException(
                    409,
                    "На агента ссылается сценарий другого агента. Сначала удалите связь.",
                )
        con.execute(
            "DELETE FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE agent_id=?)",
            (agent_id,),
        )
        con.execute("DELETE FROM sessions WHERE agent_id=?", (agent_id,))
        con.execute("DELETE FROM agents WHERE id=?", (agent_id,))
    return {"ok": True}


@app.post("/api/agents/{agent_id}/train", status_code=202)
def train(agent_id: str):
    with lock:
        row = get_agent(agent_id)
        if row["training"] == "running":
            raise HTTPException(409, "Обучение уже выполняется")
        with db() as con:
            running = con.execute(
                "SELECT COUNT(*) FROM agents WHERE training='running'"
            ).fetchone()[0]
        if running >= 2:
            raise HTTPException(429, "Уже обучаются два агента. Дождитесь завершения.")
        config = json.loads(row["config"])
        if len(config["intents"]) < 2:
            raise HTTPException(422, "Для обучения добавьте минимум два намерения")
        for rule in config["rules"]:
            if rule["action"] == "handoff":
                get_agent(rule["target_agent"])
        with db() as con:
            con.execute(
                "UPDATE agents SET training='running', error='' WHERE id=?", (agent_id,)
            )
        try:
            threading.Thread(
                target=train_worker,
                args=(agent_id, row["revision"], config),
                daemon=True,
            ).start()
        except RuntimeError as exc:
            with db() as con:
                con.execute(
                    "UPDATE agents SET training='failed', error=? WHERE id=?",
                    (str(exc), agent_id),
                )
            raise HTTPException(503, "Не удалось запустить обучение") from exc
    return {"status": "running"}


def safe_matches(pattern, text):
    try:
        return [
            m
            for m in regex.finditer(pattern, text, timeout=0.05)
            if m.start() != m.end()
        ]
    except TimeoutError as exc:
        raise HTTPException(
            422, "Слишком сложное регулярное выражение. Упростите правило сущности."
        ) from exc


@app.post("/api/agents/{agent_id}/chat")
def chat(agent_id: str, data: ChatInput):
    key = data.session_id or str(uuid.uuid4())
    with session_locks[hash(key) % len(session_locks)]:
        return process_chat(agent_id, data, key)


def process_chat(agent_id: str, data: ChatInput, session_id: str):
    agent = get_agent(agent_id)
    if not data.text.strip():
        raise HTTPException(422, "Введите сообщение")
    with db() as con:
        session = con.execute(
            "SELECT * FROM sessions WHERE id=?", (session_id,)
        ).fetchone()
    if data.session_id and not session:
        raise HTTPException(404, "Сеанс не найден. Начните новый диалог.")
    if session:
        if session["agent_id"] != agent_id:
            raise HTTPException(409, "Сеанс принадлежит другому агенту")
        state, context = session["state"], json.loads(session["context"])
        if session["active_agent"]:
            try:
                agent = get_agent(session["active_agent"])
            except HTTPException:
                raise HTTPException(409, "Активный агент диалога удалён")
    else:
        state, context = "start", {}
    original_context = context.copy()
    trace = []
    current = agent
    response = ""
    current_state = state
    for hop in range(6):
        if not current["model_path"] or not Path(current["model_path"]).is_file():
            if hop == 0:
                raise HTTPException(409, "Сначала обучите агента")
            response = "Целевой агент недоступен или ещё не обучен."
            trace.append({"agent": current["name"], "error": response})
            break
        config = json.loads(current["model_config"])
        result = parse(current["model_path"], data.text)
        intent = result.get("intent") or {}
        entities = []
        for definition in config["entities"]:
            for match in safe_matches(definition["pattern"], data.text):
                entities.append(
                    {
                        "entity": definition["name"],
                        "value": match.group(),
                        "start": match.start(),
                        "end": match.end(),
                        "extractor": "regex",
                    }
                )
        for entity in entities:
            context[entity["entity"]] = entity["value"]
        rule = (
            next(
                (
                    r
                    for r in config["rules"]
                    if r["state"] == current_state and r["intent"] == intent.get("name")
                ),
                None,
            )
            if intent.get("confidence", 0) >= config["threshold"]
            else None
        )
        step = {
            "agent": current["name"],
            "intent": intent,
            "entities": entities,
            "state": current_state,
            "action": rule["action"] if rule else "fallback",
            "model_revision": current["model_revision"],
            "stale": current["revision"] != current["model_revision"],
        }
        trace.append(step)
        if not rule:
            response = config["fallback"]
            break
        if rule["action"] != "handoff":
            response = rule["text"]
            current_state = rule["next_state"] or "start"
            step["next_state"] = current_state
            break
        step["target_agent"] = rule["target_agent"]
        try:
            target = get_agent(rule["target_agent"])
        except HTTPException:
            response = "Целевой агент недоступен или удалён."
            step["error"] = response
            break
        if not target["model_path"] or not Path(target["model_path"]).is_file():
            response = "Целевой агент недоступен или ещё не обучен."
            step["error"] = response
            break
        if hop == 5:
            response = "Превышено допустимое число передач между агентами."
            step["error"] = response
            current, current_state, context = agent, state, original_context
            break
        current = target
        current_state = "start"
    else:
        response = "Превышено допустимое число передач между агентами."
        trace.append({"error": response})
    new_state = current_state
    with db() as con:
        con.execute(
            "INSERT INTO sessions(id,agent_id,state,context,active_agent) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET state=excluded.state,context=excluded.context,active_agent=excluded.active_agent",
            (
                session_id,
                agent_id,
                new_state,
                json.dumps(context, ensure_ascii=False),
                current["id"],
            ),
        )
        con.execute(
            "INSERT INTO messages(session_id,text,response,trace) VALUES(?,?,?,?)",
            (session_id, data.text, response, json.dumps(trace, ensure_ascii=False)),
        )
    return {
        "session_id": session_id,
        "response": response,
        "trace": trace,
        "context": context,
    }


@app.get("/api/sessions/{session_id}/messages")
def messages(session_id: str):
    with db() as con:
        if not con.execute(
            "SELECT 1 FROM sessions WHERE id=?", (session_id,)
        ).fetchone():
            raise HTTPException(404, "Сеанс не найден")
        return [
            {**dict(row), "trace": json.loads(row["trace"])}
            for row in con.execute(
                "SELECT text,response,trace,created_at FROM messages WHERE session_id=? ORDER BY id",
                (session_id,),
            )
        ]


init_db()


@app.get("/static/{filename}")
def static_file(filename: str):
    if filename not in {"app.js", "style.css", "tokens.css", "base.css"}:
        raise HTTPException(404, "Файл не найден")
    return FileResponse(FRONTEND / filename)


@app.get("/")
def index():
    return FileResponse(FRONTEND / "index.html")
