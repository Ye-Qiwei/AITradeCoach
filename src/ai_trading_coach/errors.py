"""Project-level exception hierarchy."""

from __future__ import annotations


class AITradeCoachError(RuntimeError):
    """Base runtime error for AITradeCoach."""


class MissingLLMProviderError(AITradeCoachError):
    """Raised when no valid LLM provider is configured."""


class MissingAPIKeyError(AITradeCoachError):
    """Raised when the selected LLM provider key is absent."""


class LLMOutputValidationError(AITradeCoachError):
    """Raised when model JSON cannot satisfy a required schema."""


class MCPConfigurationError(AITradeCoachError):
    """Raised when MCP server/tool mapping is invalid."""


class MCPToolNotAllowedError(AITradeCoachError):
    """Raised when a tool call is blocked by allowlist."""


class ReportValidationError(AITradeCoachError):
    """Raised when report generation cannot pass judge checks."""

