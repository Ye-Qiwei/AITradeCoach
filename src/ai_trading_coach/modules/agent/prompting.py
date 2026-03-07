"""Prompt loading and message construction utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from ai_trading_coach.prompts.prompt_store import PromptStore
from ai_trading_coach.prompts.registry import PromptRegistry


@dataclass
class PromptBundle:
    prompt_name: str
    version: str
    system_prompt: str


class PromptManager:
    def __init__(self, prompt_root: str) -> None:
        self.registry = PromptRegistry(prompt_root)
        self.store = PromptStore(prompt_root)

    def load_active(self, prompt_name: str) -> PromptBundle:
        version, system_prompt = self.store.load_active(prompt_name, self.registry)
        return PromptBundle(prompt_name=prompt_name, version=version, system_prompt=system_prompt)

    @staticmethod
    def build_messages(*, system_prompt: str, payload: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False, indent=2)},
        ]
