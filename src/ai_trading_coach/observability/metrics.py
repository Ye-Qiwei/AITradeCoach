"""Metrics abstractions (placeholder)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModuleMetric:
    module_name: str
    duration_ms: int
    success: bool
