"""Configuration management."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, Field
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
    EvidenceType.PRICE_PATH.value: "yfinance:get_historical_stock_prices",
    EvidenceType.NEWS.value: "rss_search:rss_search",
    EvidenceType.FILING.value: "sec_edgar:list_filings",
    EvidenceType.MACRO.value: "fred:series_observations",
}


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore", populate_by_name=True)

    env: Literal["local", "test", "prod"] = Field(
        default="local", validation_alias=AliasChoices("ENV", "ATC_ENV")
    )
    debug: bool = Field(default=False, validation_alias=AliasChoices("DEBUG", "ATC_DEBUG"))

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
    use_gemini: bool = Field(
        default=False, validation_alias=AliasChoices("USE_GEMINI", "ATC_USE_GEMINI")
    )
    model_timeout_seconds: int = Field(
        default=20,
        validation_alias=AliasChoices("MODEL_TIMEOUT_SECONDS", "ATC_MODEL_TIMEOUT_SECONDS"),
    )

    llm_provider_name: str = Field(
        default="", validation_alias=AliasChoices("LLM_PROVIDER", "ATC_LLM_PROVIDER")
    )
    llm_model: str = Field(default="", validation_alias=AliasChoices("LLM_MODEL", "ATC_LLM_MODEL"))
    llm_timeout_seconds: float = Field(
        default=20.0,
        validation_alias=AliasChoices("LLM_TIMEOUT_SECONDS", "ATC_LLM_TIMEOUT_SECONDS"),
    )
    brave_api_key: str = Field(
        default="", validation_alias=AliasChoices("BRAVE_API_KEY", "ATC_BRAVE_API_KEY")
    )
    firecrawl_api_key: str = Field(
        default="", validation_alias=AliasChoices("FIRECRAWL_API_KEY", "ATC_FIRECRAWL_API_KEY")
    )
    agent_browser_endpoint: str = Field(
        default="",
        validation_alias=AliasChoices("AGENT_BROWSER_ENDPOINT", "ATC_AGENT_BROWSER_ENDPOINT"),
    )

    mcp_servers_json: str = Field(
        default="[]", validation_alias=AliasChoices("MCP_SERVERS", "ATC_MCP_SERVERS")
    )
    mcp_tool_allowlist_csv: str = Field(
        default="",
        validation_alias=AliasChoices("MCP_TOOL_ALLOWLIST", "ATC_MCP_TOOL_ALLOWLIST"),
    )
    evidence_tool_map_json: str = Field(
        default="",
        validation_alias=AliasChoices("EVIDENCE_TOOL_MAP", "ATC_EVIDENCE_TOOL_MAP"),
    )
    mcp_timeout_seconds: int = Field(
        default=20,
        validation_alias=AliasChoices("MCP_TIMEOUT_SECONDS", "ATC_MCP_TIMEOUT_SECONDS"),
    )
    mcp_max_retries: int = Field(
        default=1,
        validation_alias=AliasChoices("MCP_MAX_RETRIES", "ATC_MCP_MAX_RETRIES"),
    )

    agent_max_rewrite_rounds: int = Field(
        default=2,
        validation_alias=AliasChoices("AGENT_MAX_REWRITE_ROUNDS", "ATC_AGENT_MAX_REWRITE_ROUNDS"),
    )
    react_max_iterations: int = Field(
        default=6,
        validation_alias=AliasChoices("REACT_MAX_ITERATIONS", "ATC_REACT_MAX_ITERATIONS"),
    )
    react_max_tool_failures: int = Field(
        default=2,
        validation_alias=AliasChoices("REACT_MAX_TOOL_FAILURES", "ATC_REACT_MAX_TOOL_FAILURES"),
    )
    react_require_min_sources: int = Field(
        default=2,
        validation_alias=AliasChoices("REACT_REQUIRE_MIN_SOURCES", "ATC_REACT_REQUIRE_MIN_SOURCES"),
    )
    context_budget_planner: int = Field(
        default=6000,
        validation_alias=AliasChoices("CONTEXT_BUDGET_PLANNER", "ATC_CONTEXT_BUDGET_PLANNER"),
    )
    context_budget_reporter: int = Field(
        default=9000,
        validation_alias=AliasChoices("CONTEXT_BUDGET_REPORTER", "ATC_CONTEXT_BUDGET_REPORTER"),
    )
    context_budget_judge: int = Field(
        default=5000,
        validation_alias=AliasChoices("CONTEXT_BUDGET_JUDGE", "ATC_CONTEXT_BUDGET_JUDGE"),
    )

    # Backward-compatible legacy aliases.
    mcp_servers: str = "search,price,filing,news,sentiment,discussion,macro"
    enable_llm_cognition: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_LLM_COGNITION", "ATC_ENABLE_LLM_COGNITION"),
    )
    enable_llm_report: bool = Field(
        default=True,
        validation_alias=AliasChoices("ENABLE_LLM_REPORT", "ATC_ENABLE_LLM_REPORT"),
    )

    def ensure_runtime_dirs(self) -> None:
        Path(self.trace_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    def llm_provider(self) -> str:
        provider = self.llm_provider_name.strip().lower()
        if provider not in {"openai", "gemini"}:
            raise MissingLLMProviderError(
                "LLM_PROVIDER is required and must be one of: openai, gemini."
            )
        return provider

    def llm_api_key(self) -> str:
        provider = self.llm_provider()
        if provider == "openai":
            key = self.openai_api_key.strip()
            if not key:
                raise MissingAPIKeyError("OPENAI_API_KEY is required when LLM_PROVIDER=openai.")
            return key

        key = self.gemini_api_key.strip()
        if not key:
            raise MissingAPIKeyError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini.")
        return key

    def validate_llm_or_raise(self) -> None:
        self.llm_provider()
        self.llm_api_key()

    def selected_llm_model(self) -> str:
        candidate = self.llm_model.strip()
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
        payload_text = self.mcp_servers_json.strip()
        if not payload_text:
            return []
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError as exc:
            raise MCPConfigurationError(f"MCP_SERVERS must be valid JSON: {exc}") from exc
        if not isinstance(payload, list):
            raise MCPConfigurationError("MCP_SERVERS must be a JSON array.")
        try:
            return [MCPServerDefinition.model_validate(item) for item in payload]
        except Exception as exc:  # noqa: BLE001
            raise MCPConfigurationError(f"MCP_SERVERS item validation failed: {exc}") from exc

    def evidence_tool_map(self) -> dict[str, str]:
        value = self.evidence_tool_map_json.strip()
        if not value:
            return dict(DEFAULT_EVIDENCE_TOOL_MAP)
        try:
            payload = json.loads(value)
        except json.JSONDecodeError as exc:
            raise MCPConfigurationError(f"EVIDENCE_TOOL_MAP must be valid JSON: {exc}") from exc
        if not isinstance(payload, dict):
            raise MCPConfigurationError("EVIDENCE_TOOL_MAP must be a JSON object.")
        merged = dict(DEFAULT_EVIDENCE_TOOL_MAP)
        for key, mapped in payload.items():
            merged[str(key)] = str(mapped)
        return merged

    def mcp_tool_allowlist(self) -> set[str]:
        value = self.mcp_tool_allowlist_csv.strip()
        if not value:
            return set(self.evidence_tool_map().values())
        items = {chunk.strip() for chunk in value.split(",") if chunk.strip()}
        return items

    def as_mcp_settings(self) -> MCPSettings:
        return MCPSettings(
            timeout_seconds=self.mcp_timeout_seconds,
            max_retries=self.mcp_max_retries,
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

    def web_tool_status(self) -> dict[str, bool]:
        return {
            "brave_search": bool(self.brave_api_key.strip()),
            "firecrawl_extract": bool(self.firecrawl_api_key.strip()),
            "playwright_fetch": bool(self.agent_browser_endpoint.strip()),
        }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
