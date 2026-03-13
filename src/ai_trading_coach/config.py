"""Configuration management for the MVP runtime."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from ai_trading_coach.errors import MCPConfigurationError, MissingAPIKeyError, MissingLLMProviderError

PROMPT_ROOT = (Path(__file__).resolve().parents[2] / "config" / "prompts").as_posix()


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
    transport: str = "stdio"
    command: str | None = None
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict)
    url: str | None = None
    timeout_seconds: float | None = None


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        populate_by_name=True,
    )

    llm_provider_name: str = Field(default="", validation_alias="LLM_PROVIDER")
    llm_model: str = Field(default="", validation_alias="LLM_MODEL")
    openai_api_key: str = Field(default="", validation_alias="OPENAI_API_KEY")
    gemini_api_key: str = Field(default="", validation_alias="GEMINI_API_KEY")
    llm_timeout_seconds: float = Field(default=120.0, validation_alias="LLM_TIMEOUT_SECONDS")

    brave_api_key: str = Field(default="", validation_alias="BRAVE_API_KEY")
    firecrawl_api_key: str = Field(default="", validation_alias="FIRECRAWL_API_KEY")
    agent_browser_endpoint: str = Field(default="", validation_alias="AGENT_BROWSER_ENDPOINT")

    mcp_servers: list[MCPServerDefinition] = Field(default_factory=list, validation_alias="MCP_SERVERS")
    mcp_timeout_seconds: int = Field(default=60, validation_alias="MCP_TIMEOUT_SECONDS")
    mcp_max_retries: int = Field(default=1, validation_alias="MCP_MAX_RETRIES")

    agent_max_rewrite_rounds: int = Field(default=2, validation_alias="AGENT_MAX_REWRITE_ROUNDS")
    react_max_iterations: int = Field(default=6, validation_alias="REACT_MAX_ITERATIONS")
    react_max_tool_failures: int = Field(default=2, validation_alias="REACT_MAX_TOOL_FAILURES")
    react_require_min_sources: int = Field(default=2, validation_alias="REACT_REQUIRE_MIN_SOURCES")

    default_user_id: str = Field(default="demo_user", validation_alias="DEFAULT_USER_ID")
    chroma_persist_dir: str = Field(default="./.chroma", validation_alias="CHROMA_PERSIST_DIR")
    trace_output_dir: str = Field(default="./trace_logs", validation_alias="TRACE_OUTPUT_DIR")
    report_output_dir: str = Field(default="./reports", validation_alias="REPORT_OUTPUT_DIR")

    @field_validator("mcp_servers", mode="before")
    @classmethod
    def _parse_mcp_servers(cls, value: object) -> object:
        if value in (None, "", []):
            return []
        if isinstance(value, str):
            try:
                payload = json.loads(value)
            except json.JSONDecodeError as exc:
                raise MCPConfigurationError(f"MCP_SERVERS must be valid JSON: {exc}") from exc
            if not isinstance(payload, list):
                raise MCPConfigurationError("MCP_SERVERS must be a JSON array.")
            return payload
        return value

    def ensure_runtime_dirs(self) -> None:
        Path(self.trace_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.report_output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.chroma_persist_dir).mkdir(parents=True, exist_ok=True)

    @property
    def prompt_root(self) -> str:
        return PROMPT_ROOT

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
        return "gemini-2.5-pro"

    def as_memory_settings(self) -> MemorySettings:
        return MemorySettings(persist_dir=self.chroma_persist_dir)

    def mcp_server_definitions(self) -> list[MCPServerDefinition]:
        return list(self.mcp_servers)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.ensure_runtime_dirs()
    return settings
