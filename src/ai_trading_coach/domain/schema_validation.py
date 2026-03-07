"""Validation helpers for strict LLM output schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel


class StrictSchemaValidationError(ValueError):
    """Raised when a schema violates strict LLM contract requirements."""


def validate_strict_llm_schema(model_cls: type[BaseModel]) -> None:
    schema = model_cls.model_json_schema()
    if schema.get("type") != "object":
        raise StrictSchemaValidationError(f"{model_cls.__name__}: root schema type must be object")

    defs = schema.get("$defs", {})
    seen: set[int] = set()

    def resolve(node: dict[str, Any]) -> dict[str, Any]:
        if "$ref" not in node:
            return node
        ref = node["$ref"]
        if not isinstance(ref, str) or not ref.startswith("#/$defs/"):
            raise StrictSchemaValidationError(f"{model_cls.__name__}: unsupported $ref {ref!r}")
        key = ref.split("/", 2)[-1]
        target = defs.get(key)
        if not isinstance(target, dict):
            raise StrictSchemaValidationError(f"{model_cls.__name__}: unresolved $ref {ref!r}")
        return target

    def walk(node: Any, path: str) -> None:
        if isinstance(node, list):
            for idx, item in enumerate(node):
                walk(item, f"{path}[{idx}]")
            return
        if not isinstance(node, dict):
            return

        if "anyOf" in node:
            raise StrictSchemaValidationError(f"{model_cls.__name__}: anyOf is not allowed at {path}")

        resolved = resolve(node)
        obj_id = id(resolved)
        if obj_id in seen:
            return
        seen.add(obj_id)

        properties = resolved.get("properties")
        is_object = resolved.get("type") == "object" or isinstance(properties, dict)
        if is_object:
            if resolved.get("additionalProperties") is not False:
                raise StrictSchemaValidationError(
                    f"{model_cls.__name__}: additionalProperties must be false at {path}"
                )
            if not isinstance(properties, dict):
                raise StrictSchemaValidationError(f"{model_cls.__name__}: object missing properties at {path}")
            if "extensions" in properties:
                raise StrictSchemaValidationError(f"{model_cls.__name__}: extensions is forbidden at {path}")
            required = resolved.get("required")
            if not isinstance(required, list):
                raise StrictSchemaValidationError(f"{model_cls.__name__}: object missing required at {path}")
            if set(required) != set(properties.keys()):
                raise StrictSchemaValidationError(
                    f"{model_cls.__name__}: required must exactly match properties at {path}"
                )

        for key, value in resolved.items():
            if key == "extensions":
                raise StrictSchemaValidationError(f"{model_cls.__name__}: extensions is forbidden at {path}")
            walk(value, f"{path}.{key}")

    walk(schema, "$")
