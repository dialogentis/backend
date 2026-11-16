import json
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID

from pydantic import Field, StrictInt, ValidationError, field_validator, model_validator

from dialogentis.ask_slot import AskSlotDraftConfig
from dialogentis.conditions import ConditionDraftConfig, validate_draft_condition
from dialogentis.contracts import InputModel
from dialogentis.datasets.csv_import import MAX_ERRORS, MAX_FILE_BYTES, unique_object
from dialogentis.datasets.schema import Dataset, Entity, Example, Intent, validate_dataset
from dialogentis.flow_schema import Flow
from dialogentis.nlu.settings import NLUSettings, QualitySettings
from dialogentis.set_slot import SetSlotDraftConfig
from dialogentis.slots import Slot, validate_slots


class PortableConfig(InputModel):
    intents: list[Intent] = Field(max_length=100)
    entities: list[Entity]
    examples: list[Example] = Field(max_length=13000)
    slots: list[Slot]
    flow: Flow
    nlu_settings: NLUSettings = Field(default_factory=NLUSettings)
    quality_settings: QualitySettings = Field(default_factory=QualitySettings)

    @field_validator("flow", mode="before")
    @classmethod
    def check_flow_version(cls, value):
        if isinstance(value, dict):
            version = value.get("schema_version")
        elif isinstance(value, Flow):
            version = value.schema_version
        else:
            return value
        if type(version) is not int or version != 1:
            raise ValueError("Unsupported flow schema version")
        return value

    @model_validator(mode="after")
    def check_flow(self):
        _, slot_errors = validate_slots([slot.model_dump(mode="json") for slot in self.slots])
        if slot_errors:
            raise ValueError("Duplicate or invalid slots")
        allowed = {
            "start": set(),
            "switch_intent": set(),
            "fallback": {"resume_node_id", "text", "example"},
            "say": {"text"},
            "end": {"text"},
            "ask_slot": {"slot_id", "prompt", "entity_id", "max_attempts", "checks"},
            "set_slot": {"slot_id", "operation", "source", "source_slot_id", "value"},
            "condition": {"branches"},
        }
        if self.flow.layout:
            raise ValueError("Unsupported layout")
        nodes = {node.id: node for node in self.flow.nodes}
        if len(nodes) != len(self.flow.nodes):
            raise ValueError("Duplicate node")
        edge_ids: set[UUID] = set()
        handles: set[tuple[UUID, str]] = set()
        outputs = {
            "start": {"next"},
            "say": {"next"},
            "switch_intent": {"fallback", *(str(intent.id) for intent in self.intents)},
            "ask_slot": {"success", "invalid", "exhausted"},
            "set_slot": {"next", "error"},
        }
        for node in self.flow.nodes:
            if node.type not in allowed or set(node.config) - allowed[node.type]:
                raise ValueError("Unsupported node configuration")
            if "text" in node.config and (
                not isinstance(node.config["text"], str) or len(node.config["text"]) > 4000
            ):
                raise ValueError("Invalid node text")
            if "example" in node.config and (
                not isinstance(node.config["example"], str) or len(node.config["example"]) > 500
            ):
                raise ValueError("Invalid fallback example")
            if node.type == "condition":
                condition = ConditionDraftConfig.model_validate(node.config)
                validate_draft_condition(condition, {slot.id: slot for slot in self.slots})
            if node.type == "set_slot":
                assignment = SetSlotDraftConfig.model_validate(node.config)
                if any(
                    ref is not None and ref not in {slot.id for slot in self.slots}
                    for ref in (assignment.slot_id, assignment.source_slot_id)
                ):
                    raise ValueError("Unknown slot")
            if node.type == "ask_slot":
                ask = AskSlotDraftConfig.model_validate(node.config)
                if ask.slot_id is not None and ask.slot_id not in {slot.id for slot in self.slots}:
                    raise ValueError("Unknown slot")
                if ask.entity_id is not None and ask.entity_id not in {
                    entity.id for entity in self.entities
                }:
                    raise ValueError("Unknown entity")
            if "resume_node_id" in node.config:
                reference = node.config["resume_node_id"]
                try:
                    if not isinstance(reference, str) or UUID(reference) not in nodes:
                        raise ValueError("Unknown node reference")
                except ValueError as exc:
                    raise ValueError("Unknown node reference") from exc
        for edge in self.flow.edges:
            handle = (edge.source, edge.source_handle)
            if (
                edge.id in edge_ids
                or handle in handles
                or edge.source not in nodes
                or edge.target not in nodes
                or edge.source_handle
                not in (
                    {
                        "else",
                        *(
                            str(branch.id)
                            for branch in ConditionDraftConfig.model_validate(
                                nodes[edge.source].config
                            ).branches
                        ),
                    }
                    if nodes[edge.source].type == "condition"
                    else outputs.get(nodes[edge.source].type, set())
                )
            ):
                raise ValueError("Invalid edge")
            edge_ids.add(edge.id)
            handles.add(handle)
        return self


class PortableFile(InputModel):
    schema_version: StrictInt
    config: PortableConfig

    @field_validator("schema_version")
    @classmethod
    def check_version(cls, value: int) -> int:
        if value != 1:
            raise ValueError("Unsupported schema version")
        return value


@dataclass(frozen=True)
class JsonIssue:
    path: tuple[str | int, ...]
    code: str


@dataclass(frozen=True)
class JsonPreview:
    config: PortableConfig | None
    errors: list[JsonIssue]
    warnings: list[JsonIssue] = field(default_factory=list)


def _safe_path(path: tuple) -> tuple[str | int, ...]:
    fields = {
        "schema_version",
        "config",
        "intents",
        "entities",
        "examples",
        "slots",
        "flow",
        "nlu_settings",
        "score",
        "margin",
        "similarity",
        "quality_settings",
        "intent_f1",
        "entity_f1",
        "id",
        "code",
        "name",
        "description",
        "enabled",
        "value_type",
        "required",
        "sensitive",
        "default",
        "method",
        "dictionary",
        "value",
        "synonyms",
        "pattern",
        "group",
        "text",
        "intent_id",
        "expected_fallback",
        "split",
        "spans",
        "entity_id",
        "start",
        "end",
        "flow_id",
        "nodes",
        "edges",
        "layout",
        "type",
        "position",
        "x",
        "y",
        "source",
        "source_handle",
        "target",
    }
    return tuple(part if isinstance(part, int) or part in fields else "<unknown>" for part in path)


def _reject_constant(value: str) -> None:
    raise ValueError("Invalid JSON constant")


def parse_json(content: bytes) -> tuple[PortableConfig | None, list[JsonIssue]]:
    if len(content) > MAX_FILE_BYTES:
        return None, [JsonIssue((), "file_too_large")]
    try:
        data = json.loads(
            content.decode("utf-8-sig"),
            object_pairs_hook=unique_object,
            parse_constant=_reject_constant,
        )
        json.dumps(data, ensure_ascii=False, allow_nan=False).encode("utf-8")
    except UnicodeError, ValueError, RecursionError:
        return None, [JsonIssue((), "invalid_json")]
    try:
        file = PortableFile.model_validate(data)
    except ValidationError as exc:
        errors = exc.errors(include_input=False, include_context=False, include_url=False)
        result = [
            JsonIssue(_safe_path(error["loc"]), error["type"]) for error in errors[:MAX_ERRORS]
        ]
        if len(errors) > MAX_ERRORS:
            result.append(JsonIssue((), "too_many_errors"))
        return None, result
    return file.config, []


def _dataset(config: PortableConfig) -> Dataset:
    return Dataset(intents=config.intents, entities=config.entities, examples=config.examples)


def preview_json(
    imported: PortableConfig, current: PortableConfig, mode: Literal["merge", "replace"]
) -> JsonPreview:
    if mode not in {"merge", "replace"}:
        raise ValueError("Invalid import mode")
    try:
        imported = PortableConfig.model_validate(imported.model_dump())
        current = PortableConfig.model_validate(current.model_dump())
    except ValidationError:
        return JsonPreview(None, [JsonIssue(("config",), "invalid_configuration")])
    warnings = []
    if mode == "merge":
        if imported.flow != current.flow or imported.slots != current.slots:
            warnings.append(JsonIssue(("config", "flow"), "flow_not_imported"))
        if (imported.nlu_settings, imported.quality_settings) != (
            current.nlu_settings,
            current.quality_settings,
        ):
            warnings.append(JsonIssue(("config",), "settings_not_imported"))
        try:
            dataset = Dataset(
                intents=[*current.intents, *imported.intents],
                entities=[*current.entities, *imported.entities],
                examples=[*current.examples, *imported.examples],
            )
        except ValidationError:
            return JsonPreview(None, [JsonIssue(("config",), "dataset_limit")], warnings)
        candidate = PortableConfig(
            intents=dataset.intents,
            entities=dataset.entities,
            examples=dataset.examples,
            slots=current.slots,
            flow=current.flow,
            nlu_settings=current.nlu_settings,
            quality_settings=current.quality_settings,
        )
    else:
        candidate = imported
    issues, _ = validate_dataset(_dataset(candidate))
    readiness = {
        "insufficient_intents",
        "insufficient_intent_examples",
        "insufficient_entity_examples",
    }
    errors = [
        JsonIssue(("config", *issue.path), issue.code)
        for issue in issues
        if issue.code not in readiness
    ]
    try:
        encoded = json.dumps(
            candidate.model_dump(mode="json"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
    except ValueError, UnicodeError:
        return JsonPreview(None, [JsonIssue(("config",), "invalid_configuration")], warnings)
    if len(encoded) > MAX_FILE_BYTES:
        errors.append(JsonIssue(("config",), "configuration_too_large"))
    if len(errors) > MAX_ERRORS:
        errors = [*errors[:MAX_ERRORS], JsonIssue((), "too_many_errors")]
    return JsonPreview(None if errors else candidate, errors, warnings)


def export_json(config: PortableConfig) -> bytes:
    try:
        config = PortableConfig.model_validate(config.model_dump())
    except ValidationError as exc:
        raise ValueError("Invalid configuration") from exc
    preview = preview_json(config, config, "replace")
    if preview.errors:
        raise ValueError("Invalid configuration")
    try:
        content = json.dumps(
            PortableFile(schema_version=1, config=config).model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
        ).encode("utf-8")
    except (ValueError, UnicodeEncodeError) as exc:
        raise ValueError("Invalid configuration") from exc
    if len(content) > MAX_FILE_BYTES:
        raise ValueError("Export exceeds file limit")
    return content
