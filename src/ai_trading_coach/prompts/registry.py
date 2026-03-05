"""Prompt registry with version discovery and active-version management."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


PROMPT_FILE_RE = re.compile(r"^(?P<name>[a-z0-9_]+)\.(?P<version>v\d+)\.md$")


@dataclass(frozen=True)
class PromptVersionRef:
    prompt_name: str
    version: str
    path: Path


class PromptRegistry:
    def __init__(self, root_dir: str, manifest_file: str = "manifest.json") -> None:
        self.root = Path(root_dir)
        self.manifest_path = self.root / manifest_file

    def load(self, prompt_file: str) -> str:
        path = self.root / prompt_file
        if not path.exists():
            raise FileNotFoundError(f"Prompt file not found: {path}")
        return path.read_text(encoding="utf-8")

    def list_prompt_names(self) -> list[str]:
        names = {ref.prompt_name for ref in self._discover_prompt_refs()}
        return sorted(names)

    def list_versions(self, prompt_name: str) -> list[str]:
        refs = [ref for ref in self._discover_prompt_refs() if ref.prompt_name == prompt_name]
        return [ref.version for ref in sorted(refs, key=lambda item: self._version_rank(item.version))]

    def load_version(self, prompt_name: str, version: str) -> str:
        filename = f"{prompt_name}.{version}.md"
        return self.load(filename)

    def get_active_version(self, prompt_name: str) -> str | None:
        manifest = self._load_manifest()
        version = manifest.get("active_versions", {}).get(prompt_name)
        if isinstance(version, str) and version:
            return version

        versions = self.list_versions(prompt_name)
        return versions[-1] if versions else None

    def load_active(self, prompt_name: str) -> tuple[str, str]:
        version = self.get_active_version(prompt_name)
        if version is None:
            raise FileNotFoundError(f"No prompt versions found for {prompt_name}")
        return version, self.load_version(prompt_name, version)

    def set_active_version(self, prompt_name: str, version: str) -> None:
        versions = self.list_versions(prompt_name)
        if version not in versions:
            raise ValueError(f"Prompt version not found: {prompt_name}.{version}")

        manifest = self._load_manifest()
        active_versions = manifest.setdefault("active_versions", {})
        active_versions[prompt_name] = version
        self.manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _discover_prompt_refs(self) -> list[PromptVersionRef]:
        refs: list[PromptVersionRef] = []
        if not self.root.exists():
            return refs

        for path in self.root.glob("*.md"):
            match = PROMPT_FILE_RE.match(path.name)
            if match is None:
                continue
            refs.append(
                PromptVersionRef(
                    prompt_name=match.group("name"),
                    version=match.group("version"),
                    path=path,
                )
            )
        return refs

    def _load_manifest(self) -> dict[str, object]:
        if not self.manifest_path.exists():
            return {"active_versions": {}}
        try:
            loaded = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                return loaded
        except json.JSONDecodeError:
            pass
        return {"active_versions": {}}

    def _version_rank(self, version: str) -> tuple[int, str]:
        if len(version) <= 1 or not version.startswith("v"):
            return (0, version)
        try:
            return (int(version[1:]), version)
        except ValueError:
            return (0, version)
