"""Deterministic ID generation helpers for pipeline entities."""

from __future__ import annotations

import hashlib
import re


_WS_RE = re.compile(r"\s+")


def _normalize(value: str) -> str:
    return _WS_RE.sub(" ", value.strip().lower())


def _short_hash(*parts: str) -> str:
    payload = "|".join(_normalize(part) for part in parts)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def make_parse_id(run_id: str, raw_log_text: str) -> str:
    return f"parse_{_short_hash(run_id, raw_log_text)}"


def make_action_id(run_id: str, ordinal: int, action: object) -> str:
    action_name = getattr(action, "action", "")
    target_asset = getattr(action, "target_asset", "")
    return f"act_{_short_hash(run_id, str(ordinal), str(action_name), str(target_asset))}"


def make_judgement_id(run_id: str, judgement_kind: str, ordinal: int, category: str, target: str, thesis: str) -> str:
    return f"jdg_{_short_hash(run_id, judgement_kind, str(ordinal), category, target, thesis)}"


def make_research_id(run_id: str) -> str:
    return f"research_{_short_hash(run_id)}"
