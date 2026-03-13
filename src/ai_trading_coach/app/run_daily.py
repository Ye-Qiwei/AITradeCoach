"""Scheduled trigger entrypoint."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest


def build_scheduled_request(log_path: str, user_id: str | None = None) -> ReviewRunRequest:
    settings = get_settings()
    owner = user_id or settings.default_user_id
    content = Path(log_path).read_text(encoding="utf-8")
    today = date.today()
    return ReviewRunRequest(
        run_id=f"scheduled_{owner}_{today.isoformat()}",
        user_id=owner,
        run_date=today,
        trigger_type=TriggerType.SCHEDULED,
        raw_log_text=content,
        options={"dry_run": False, "debug_mode": False},
    )
