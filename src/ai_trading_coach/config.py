"""Configuration management."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_trading_coach.domain.enums import EvidenceType, ModuleName
from ai_trading_coach.errors import MCPConfigurationError, MissingAPIKeyError, MissingLLMProviderError


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


class MCPServerDefinition(BaseModel):
    server_id: str
    transport: Literal["stdio", "http", "sse"] = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    timeout_seconds: float | None = None


class MCPSettings(BaseModel):
    timeout_seconds: int = 20
    max_retries: int = 1
    servers: list[MCPServerDefinition] = Field(default_factory=list)
    tool_allowlist: set[str] = Field(default_factory=set)


class PathSettings(BaseModel):
    trace_output_dir: str = "./trace_logs"
    report_output_dir: str = "./reports"
    prompt_registry_path: str = "./config/prompts"


DEFAULT_EVIDENCE_TOOL_MAP: dict[str, str] = {
    EvidenceType.PRICE_PATH.value: "yahoo_finance:price_history",
    EvidenceType.NEWS.value: "rss_search:rss_search",
    EvidenceType.FILING.value: "sec_edgar:list_filings",
    EvidenceType.MACRO.value: "fred:series_observations",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    atc_env: Literal["local", "test", "prod"] = "local"
    atc_debug: bool = False

    gemini_api_key: str = ""
    openai_api_key: str = ""
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

    scheduler_cron: str = "0 20 * * 1-5"
    default_user_id: str = "demo_user"

    trace_output_dir: str = "./trace_logs"
    report_output_dir: str = "./reports"
    log_level: str = "INFO"

    prompt_version: str = "agent_v2"
    prompt_registry_path: str = "./config/prompts"
    atc_use_gemini: bool = False
    atc_model_timeout_seconds: int = 20

    atc_llm_provider: str = ""
    atc_llm_model: str = ""
    atc_llm_timeout_seconds: float = 20.0

    atc_mcp_servers: str = "[]"
    atc_mcp_tool_allowlist: str = ""
    atc_evidence_tool_map: str = ""
    atc_mcp_timeout_seconds: int = 20
    atc_mcp_max_retries: int = 1

    atc_agent_max_rewrite_rounds: int = 2
    atc_context_budget_planner: int = 6000
    atc_context_budget_reporter: int = 9000
    atc_context_budget_judge: int = 5000

    # Backward-compatible legacy aliases.
    mcp_timeout_seconds: int = 12
    mcp_max_retries: int = 2
    mcp_servers: str = "search,price,filing,news,sentiment,discussion,macro"
    atc_enable_llm_cognition: bool = True
    atc_enable_llm_report: bool = True

    def ensure_runtime_dirs(self) -> None:
        Path(self.trace_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    def llm_provider(self) -> str:
        provider = self.atc_llm_provider.strip().lower()
        if provider not in {"openai", "gemini"}:
            raise MissingLLMProviderError(
                "ATC_LLM_PROVIDER is required and must be one of: openai, gemini."
            )
        return provider

    def llm_api_key(self) -> str:
        provider = self.llm_provider()
        if provider == "openai":
            key = self.openai_api_key.strip()
            if not key:
                raise MissingAPIKeyError("OPENAI_API_KEY is required when ATC_LLM_PROVIDER=openai.")
            return key

        key = self.gemini_api_key.strip()
        if not key:
            raise MissingAPIKeyError("GEMINI_API_KEY is required when ATC_LLM_PROVIDER=gemini.")
        return key

    def validate_llm_or_raise(self) -> None:
        self.llm_provider()
        self.llm_api_key()

    def selected_llm_model(self) -> str:
        candidate = self.atc_llm_model.strip()
        if candidate:
            return candidate
        if self.llm_provider() == "openai":
            return "gpt-4o-mini"
        return self.gemini_model

    def as_model_settings(self) -> ModelSettings:
        return ModelSettings(provider=self.llm_provider(), model_name=self.selected_llm_model())

    def as_memory_settings(self) -> MemorySettings:
        return MemorySettings(persist_dir=self.chroma_persist_dir)

    def mcp_server_definitions(self) -> list[MCPServerDefinition]:
        payload_text = self.atc_mcp_servers.strip()
        if not payload_text:
            return []
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise MCPConfigurationError(f"ATC_MCP_SERVERS must be valid JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise MCPConfigurationError("ATC_MCP_SERVERS must be a JSON array.")
        try:
            return [MCPServerDefinition.model_validate(item) for item in payload]
        except Exception as exc:  # noqa: BLE001
            raise MCPConfigurationError(f"ATC_MCP_SERVERS item validation failed: {exc}") from exc

    def evidence_tool_map(self) -> dict[str, str]:
        value = self.atc_evidence_tool_map.strip()
        if not value:
            return dict(DEFAULT_EVIDENCE_TOOL_MAP)
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MCPConfigurationError(f"ATC_EVIDENCE_TOOL_MAP must be valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise MCPConfigurationError("ATC_EVIDENCE_TOOL_MAP must be a JSON object.")
        merged = dict(DEFAULT_EVIDENCE_TOOL_MAP)
        for key, mapped in payload.items():
            merged[str(key)] = str(mapped)
        return merged

    def mcp_tool_allowlist(self) -> set[str]:
        value = self.atc_mcp_tool_allowlist.strip()
        if not value:
            return set(self.evidence_tool_map().values())
        items = {chunk.strip() for chunk in value.split(",") if chunk.strip()}
        return items

    def as_mcp_settings(self) -> MCPSettings:
        return MCPSettings(
            timeout_seconds=self.atc_mcp_timeout_seconds,
            max_retries=self.atc_mcp_max_retries,
            servers=self.mcp_server_definitions(),
            tool_allowlist=self.mcp_tool_allowlist(),
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
        return candidate or self.selected_llm_model()

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
