"""Manual trigger entrypoint for local development."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import typer

from ai_trading_coach.app.factory import build_pipeline_orchestrator
from ai_trading_coach.config import get_settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.modules.agent.research_tools import resolve_research_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.observability.tracing import save_run_trace

app = typer.Typer(add_completion=False)


def _environment_report() -> dict[str, Any]:
    settings = get_settings()
    warnings: list[str] = []
    errors: list[str] = []
    llm: dict[str, Any] = {"ok": False, "provider": "", "model": "", "error": None}
    tool_diagnostics: list[dict[str, Any]] = []

    try:
        provider = settings.llm_provider()
        settings.llm_api_key()
        llm = {
            "ok": True,
            "provider": provider,
            "model": settings.selected_llm_model(),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        llm["error"] = str(exc)
        errors.append(str(exc))

    mcp: dict[str, Any] = {
        "ok": False,
        "server_count": 0,
        "error": None,
    }
    try:
        manager = MCPClientManager(settings=settings)
        resolved = resolve_research_tools(settings=settings, mcp_manager=manager)
        tool_diagnostics = [
            {
                "agent_name": item.spec.name,
                "backend": item.spec.backend_ref,
                "backend_kind": item.spec.backend_kind,
                "capability_group": item.spec.capability_group,
                "available": item.available,
                "reason": item.reason,
            }
            for item in resolved
        ]
        mcp = {
            "ok": True,
            "server_count": len(manager.server_map),
            "diagnostics": manager.diagnostics(),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        mcp["error"] = str(exc)
        errors.append(str(exc))

    enabled = [item for item in tool_diagnostics if item["available"]]
    disabled = [item for item in tool_diagnostics if not item["available"]]
    if not enabled:
        warnings.append(
            "No research tools are configured. Add MCP_SERVERS or enable "
            "BRAVE_API_KEY / FIRECRAWL_API_KEY / AGENT_BROWSER_ENDPOINT."
        )
    if not any(item["capability_group"] == "general_web" for item in enabled):
        warnings.append(
            "General web research tools are disabled. Dynamic-page and broad web lookup fallbacks will be unavailable."
        )

    status = "error" if errors else "warn" if warnings else "ok"
    return {
        "status": status,
        "llm": llm,
        "mcp": mcp,
        "tools": tool_diagnostics,
        "agent_tools": sorted({item["agent_name"] for item in enabled}),
        "skipped_tools": disabled,
        "warnings": warnings,
        "errors": errors,
    }


def _ensure_research_tools_configured_or_raise() -> None:
    report = _environment_report()
    if report["errors"]:
        typer.echo(f"configuration_error: {report['errors'][0]}", err=True)
        raise typer.Exit(code=2)
    if not report["agent_tools"]:
        typer.echo(
            "configuration_error: No research tools are configured. Run "
            "`python3 -m ai_trading_coach.app.run_manual doctor` and configure "
            "MCP_SERVERS and/or BRAVE_API_KEY, FIRECRAWL_API_KEY, "
            "AGENT_BROWSER_ENDPOINT.",
            err=True,
        )
        raise typer.Exit(code=2)


@app.command()
def run(
    user_id: str = typer.Option("demo_user", help="User ID"),
    log_file: str = typer.Option(
        "examples/logs/daily_log_sample.md", help="Path to markdown log"
    ),
    run_date: str = typer.Option(date.today().isoformat(), help="Run date YYYY-MM-DD"),
    dry_run: bool = typer.Option(False, help="Disable memory + report/trace writes"),
) -> None:
    settings = get_settings()
    settings.validate_llm_or_raise()
    _ensure_research_tools_configured_or_raise()
    text = Path(log_file).read_text(encoding="utf-8")
    orchestrator = build_pipeline_orchestrator(settings)

    request = ReviewRunRequest(
        run_id=f"manual_{user_id}_{run_date}",
        user_id=user_id,
        run_date=date.fromisoformat(run_date),
        trigger_type=TriggerType.MANUAL,
        raw_log_text=text,
        options={"dry_run": dry_run, "debug_mode": False},
    )

    result = orchestrator.run(request)
    if not dry_run and result.report is not None:
        report_path = Path(settings.report_output_dir) / f"{result.run_id}.md"
        report_path.write_text(result.report.markdown_body, encoding="utf-8")
    if not dry_run and result.trace is not None:
        save_run_trace(result.trace, settings.trace_output_dir)
    typer.echo(result.model_dump_json(indent=2))


@app.command()
def doctor(
    as_json: bool = typer.Option(False, "--json", help="Output full report as JSON"),
    strict: bool = typer.Option(
        False, help="Exit with code 1 on warnings as well as errors"
    ),
) -> None:
    report = _environment_report()
    if as_json:
        typer.echo(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        typer.echo(f"status: {report['status']}")
        llm = report["llm"]
        typer.echo(
            f"llm: {'ok' if llm['ok'] else 'error'}"
            f" provider={llm['provider'] or '-'}"
            f" model={llm['model'] or '-'}"
        )
        if llm["error"]:
            typer.echo(f"llm_error: {llm['error']}")

        mcp = report["mcp"]
        configured_mcp = len(
            [
                item
                for item in report["tools"]
                if item["available"] and item["backend_kind"] == "mcp"
            ]
        )
        typer.echo(
            f"mcp: {'ok' if mcp['ok'] else 'error'}"
            f" servers={mcp['server_count']}"
            f" configured_tools={configured_mcp}"
        )
        for item in report["tools"]:
            line = (
                f"tool: {item['agent_name']} -> {item['backend']} "
                f"[{item['backend_kind']}/{item['capability_group']}] "
                f"({'enabled' if item['available'] else 'skipped'})"
            )
            if item["reason"]:
                line += f" reason={item['reason']}"
            typer.echo(line)
        if mcp["error"]:
            typer.echo(f"mcp_error: {mcp['error']}")

        enabled_web = [
            item["agent_name"]
            for item in report["tools"]
            if item["available"] and item["capability_group"] == "general_web"
        ]
        disabled_web = [
            item["agent_name"]
            for item in report["tools"]
            if (not item["available"]) and item["capability_group"] == "general_web"
        ]
        typer.echo(f"web_tools_enabled: {', '.join(enabled_web) if enabled_web else '-'}")
        typer.echo(f"web_tools_disabled: {', '.join(disabled_web) if disabled_web else '-'}")
        typer.echo(
            f"agent_tools: {', '.join(report['agent_tools']) if report['agent_tools'] else '-'}"
        )

        for warning in report["warnings"]:
            typer.echo(f"warning: {warning}")
        for error in report["errors"]:
            typer.echo(f"error: {error}")

    if report["errors"] or (strict and report["warnings"]):
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
