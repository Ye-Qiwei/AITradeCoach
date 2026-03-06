from __future__ import annotations

import pytest

from ai_trading_coach.app.factory import build_pipeline_orchestrator
from ai_trading_coach.config import Settings
from ai_trading_coach.errors import MissingAPIKeyError, MissingLLMProviderError


def test_missing_llm_provider_raises_immediately() -> None:
    settings = Settings(_env_file=None, llm_provider_name="")
    with pytest.raises(MissingLLMProviderError):
        settings.validate_llm_or_raise()


def test_missing_api_key_raises_before_orchestrator_build() -> None:
    settings = Settings(_env_file=None, llm_provider_name="openai", openai_api_key="")
    with pytest.raises(MissingAPIKeyError):
        build_pipeline_orchestrator(settings)

