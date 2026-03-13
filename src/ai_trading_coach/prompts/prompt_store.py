"""Prompt loading + learned overlay persistence."""

from __future__ import annotations

import json
from pathlib import Path


class PromptStore:
    def __init__(self, root_dir: str, overlay_file: str = "learned_overlays.json") -> None:
        self.root = Path(root_dir)
        self.overlay_path = self.root / overlay_file

    def load_active(self, name: str) -> str:
        path = self.root / f"{name}.md"
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        base = path.read_text(encoding="utf-8")
        overlays = self._load_overlays().get(name, [])
        if not overlays:
            return base
        learned = "\n".join(f"- {entry['instruction']}" for entry in overlays)
        return f"{base}\n\n# Learned overlay instructions\n{learned}"

    def append_overlay(self, prompt_name: str, instruction: str, reason: str) -> None:
        payload = self._load_overlays()
        payload.setdefault(prompt_name, []).append({"instruction": instruction, "reason": reason})
        self.root.mkdir(parents=True, exist_ok=True)
        self.overlay_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load_overlays(self) -> dict:
        if not self.overlay_path.exists():
            return {}
        return json.loads(self.overlay_path.read_text(encoding="utf-8"))
