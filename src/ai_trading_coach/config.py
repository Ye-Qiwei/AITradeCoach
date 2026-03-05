"""Configuration management."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_trading_coach.domain.enums import ModuleName


class ModelSettings(BaseModel):
    provider: str = "gemini"
    model_name: str = "gemini-2.5-pro"
    temperature: float = 0.1
    max_output_tokens: int = 4096


class MemorySettings(BaseModel):
    provider: str = "chromadb"
    persist_dir: str = "./.chroma"
    raw_logs_collection: str = "raw_logs"
    cognitive_cases_collection: str = "cognitive_cases"
    user_profile_collection: str = "user_profile"
    active_theses_collection: str = "active_theses"
    agent_improvement_notes_collection: str = "agent_improvement_notes"


class MCPSettings(BaseModel):
    timeout_seconds: int = 12
    max_retries: int = 2
    servers: list[str] = Field(default_factory=lambda: ["search", "price", "filing", "news"])


class PathSettings(BaseModel):
    trace_output_dir: str = "./trace_logs"
    report_output_dir: str = "./reports"
    prompt_registry_path: str = "./config/prompts"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    atc_env: Literal["local", "test", "prod"] = "local"
    atc_debug: bool = False

    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-pro"
    model_default: str = ""
    model_log_understanding: str = ""
    model_cognition_extraction: str = ""
    model_evidence_planning: str = ""
    model_window_selection: str = ""
    model_cognition_evaluation: str = ""
    model_report_generation: str = ""
    model_promptops: str = ""

    chroma_persist_dir: str = "./.chroma"
    chroma_tenant: str = "default_tenant"
    chroma_database: str = "default_database"

    mcp_timeout_seconds: int = 12
    mcp_max_retries: int = 2
    mcp_servers: str = "search,price,filing,news,sentiment,discussion,macro"

    scheduler_cron: str = "0 20 * * 1-5"
    default_user_id: str = "demo_user"

    trace_output_dir: str = "./trace_logs"
    report_output_dir: str = "./reports"
    log_level: str = "INFO"

    prompt_version: str = "baseline_v1"
    prompt_registry_path: str = "./config/prompts"
    atc_use_gemini: bool = False
    atc_model_timeout_seconds: int = 20

    def ensure_runtime_dirs(self) -> None:
        Path(self.trace_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    def as_model_settings(self) -> ModelSettings:
        return ModelSettings(model_name=self.default_model())

    def as_memory_settings(self) -> MemorySettings:
        return MemorySettings(persist_dir=self.chroma_persist_dir)

    def as_mcp_settings(self) -> MCPSettings:
        servers = [x.strip() for x in self.mcp_servers.split(",") if x.strip()]
        return MCPSettings(
            timeout_seconds=self.mcp_timeout_seconds,
            max_retries=self.mcp_max_retries,
            servers=servers,
        )

    def as_path_settings(self) -> PathSettings:
        return PathSettings(
            trace_output_dir=self.trace_output_dir,
            report_output_dir=self.report_output_dir,
            prompt_registry_path=self.prompt_registry_path,
        )

    def default_model(self) -> str:
        return self.model_default or self.gemini_model

    def model_for_module(self, module: ModuleName | str) -> str:
        name = module.value if isinstance(module, ModuleName) else str(module)
        mapping = {
            ModuleName.LOG_INTAKE.value: self.model_log_understanding,
            ModuleName.COGNITION_ENGINE.value: self.model_cognition_extraction,
            ModuleName.EVIDENCE_PLANNER.value: self.model_evidence_planning,
            ModuleName.WINDOW_SELECTOR.value: self.model_window_selection,
            ModuleName.EVALUATOR.value: self.model_cognition_evaluation,
            ModuleName.REPORT_GENERATOR.value: self.model_report_generation,
            ModuleName.PROMPTOPS.value: self.model_promptops,
        }
        candidate = mapping.get(name, "")
        return candidate or self.default_model()

    def module_model_map(self) -> dict[str, str]:
        return {
            ModuleName.LOG_INTAKE.value: self.model_for_module(ModuleName.LOG_INTAKE),
            ModuleName.COGNITION_ENGINE.value: self.model_for_module(ModuleName.COGNITION_ENGINE),
            ModuleName.EVIDENCE_PLANNER.value: self.model_for_module(ModuleName.EVIDENCE_PLANNER),
            ModuleName.WINDOW_SELECTOR.value: self.model_for_module(ModuleName.WINDOW_SELECTOR),
            ModuleName.EVALUATOR.value: self.model_for_module(ModuleName.EVALUATOR),
            ModuleName.REPORT_GENERATOR.value: self.model_for_module(ModuleName.REPORT_GENERATOR),
            ModuleName.PROMPTOPS.value: self.model_for_module(ModuleName.PROMPTOPS),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
