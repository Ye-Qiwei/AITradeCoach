"""Prompt loading and message construction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_trading_coach.prompts.prompt_store import PromptStore


@dataclass
class PromptBundle:
    prompt_name: str
    system_prompt: str


class PromptManager:
    def __init__(self, prompt_root: str) -> None:
        self.store = PromptStore(prompt_root)

    def load_active(self, prompt_name: str) -> PromptBundle:
        return PromptBundle(prompt_name=prompt_name, system_prompt=self.store.load_active(prompt_name))

    @staticmethod
    def _render_markdown(value: Any, heading: str | None = None) -> str:
        lines: list[str] = []
        if heading:
            lines.append(f"## {heading}")
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (dict, list)):
                    lines.append(f"### {key}")
                    lines.append(PromptManager._render_markdown(item))
                else:
                    lines.append(f"- {key}: {item}")
            return "\n".join(lines).strip()
        if isinstance(value, list):
            if value and all(isinstance(row, dict) for row in value):
                keys = sorted({str(k) for row in value for k in row.keys()})
                if keys:
                    lines.append("| " + " | ".join(keys) + " |")
                    lines.append("|" + "|".join(["---"] * len(keys)) + "|")
                    for row in value:
                        lines.append("| " + " | ".join(str(row.get(key, "")) for key in keys) + " |")
                    return "\n".join(lines).strip()
            for item in value:
                if isinstance(item, (dict, list)):
                    lines.append(f"-\n{PromptManager._render_markdown(item)}")
                else:
                    lines.append(f"- {item}")
            return "\n".join(lines).strip()
        lines.append(str(value))
        return "\n".join(lines).strip()

    @staticmethod
    def build_messages(*, system_prompt: str, context: dict[str, Any]) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": PromptManager._render_markdown(context, heading="Task Context")},
        ]
