"""Train one snapshot; parent owns authorization, deadlines and persistence."""

import json
import os
import sys
import threading
from dataclasses import asdict
from uuid import UUID

from pydantic import SecretStr

from dialogentis.config import Settings
from dialogentis.datasets.schema import Dataset
from dialogentis.nlu.adapter import NLUAdapter
from dialogentis.nlu.intent import Thresholds
from dialogentis.nlu.model_store import ModelIdentity, pack_model
from dialogentis.nlu.settings import QualitySettings

MAX_REQUEST_BYTES = 12 * 1024 * 1024
MAX_REPORT_BYTES = 32 * 1024 * 1024


def stage(name: str) -> None:
    print(name, flush=True)


def _watch_parent() -> None:
    if not os.read(sys.stdin.fileno(), 1):
        os._exit(73)


def _write(fd: int, content: bytes) -> None:
    with os.fdopen(fd, "wb") as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())


def main() -> None:
    line = sys.stdin.buffer.readline(MAX_REQUEST_BYTES + 1)
    if len(line) > MAX_REQUEST_BYTES or not line.endswith(b"\n"):
        raise ValueError("Invalid training request")
    threading.Thread(target=_watch_parent, daemon=True).start()
    request = json.loads(line)
    stage("validation")
    snapshot = request["snapshot"]
    if snapshot["schema_version"] != 1:
        raise ValueError("Unsupported snapshot")
    config = snapshot["config"]
    dataset = Dataset.model_validate(
        {key: config[key] for key in ("intents", "entities", "examples")}
    )
    thresholds = Thresholds(**snapshot["nlu_settings"])
    quality = QualitySettings.model_validate(config.get("quality_settings", {}))
    identity = ModelIdentity(
        *(
            UUID(request[key])
            for key in ("organization_id", "agent_id", "version_id", "artifact_id")
        )
    )
    adapter = NLUAdapter.train(identity.version_id, dataset, thresholds, on_stage=stage)
    stage("evaluation")
    report = adapter.evaluate(dataset)
    example = next(item for item in dataset.examples if item.split == "train")
    if not adapter.predict(example.text).intent.ranking:
        raise ValueError("Model smoke check failed")
    warnings = []
    if report.intents.macro_f1 is None:
        warnings.append("intent_quality_unavailable")
    elif report.intents.macro_f1 < quality.intent_f1:
        warnings.append("low_intent_quality")
    if report.entities.f1 is None:
        warnings.append("entity_quality_unavailable")
    elif report.entities.f1 < quality.entity_f1:
        warnings.append("low_entity_quality")
    if report.intents.unknown_rejection is None:
        warnings.append("unknown_quality_unavailable")
    if any(item.support < 3 for item in report.intents.per_intent):
        warnings.append("few_test_examples")
    result = {
        "version_id": str(identity.version_id),
        "training_examples": adapter.intent_model.manifest["training_examples"],
        "intents": asdict(report.intents),
        "entities": asdict(report.entities),
        "graph_validation": "not_checked",
        "warnings": warnings,
    }
    report_bytes = json.dumps(
        result, ensure_ascii=False, allow_nan=False, default=str, separators=(",", ":")
    ).encode("utf-8")
    if len(report_bytes) > MAX_REPORT_BYTES:
        raise ValueError("Training report exceeds size limit")
    stage("packaging")
    settings = Settings(
        _env_file=None,
        encryption_key=SecretStr(os.environ["DIALOGENTIS_TRAINING_KEY"]),
        encryption_key_version=int(os.environ["DIALOGENTIS_TRAINING_KEY_VERSION"]),
        max_model_artifact_bytes=int(os.environ["DIALOGENTIS_TRAINING_MODEL_LIMIT"]),
    )
    model = pack_model(adapter, identity, settings)
    _write(int(os.environ["DIALOGENTIS_TRAINING_MODEL_FD"]), model)
    _write(int(os.environ["DIALOGENTIS_TRAINING_REPORT_FD"]), report_bytes)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        os._exit(1)
