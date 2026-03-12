#!/usr/bin/env python3
"""Run the real LangGraph review pipeline and inspect node-by-node updates.

This debugger keeps the *real* pipeline behavior:
- real prompts
- real LLM API calls
- real MCP / web tools
- real LangGraph topology and routing

It adds rich observability for debugging:
- captures full LLM inputs / outputs for parser, reporter, judge
- captures real research-agent invoke payload / result for execute_collection
- saves per-node artifacts to disk in readable markdown + JSON
- prints concise terminal summaries while preserving full outputs on disk

Usage examples
--------------
Run until a specific node executes once::

    python scripts/debug_langgraph_nodes.py --node parse_log
    python scripts/debug_langgraph_nodes.py --node execute_collection

Run the full graph and print every node update::

    python scripts/debug_langgraph_nodes.py --node all

Stop after the second execute_collection (useful when research loops)::

    python scripts/debug_langgraph_nodes.py --node execute_collection --occurrence 2

Also print full LLM payloads inline::

    python scripts/debug_langgraph_nodes.py --node parse_log --print-llm-full
"""

from __future__ import annotations

import argparse
import inspect
import json
import sys
import textwrap
import traceback
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from time import perf_counter
from types import MethodType
from typing import Any, Callable, Iterable

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ai_trading_coach.app.factory import build_orchestrator_modules
from ai_trading_coach.config import Settings, get_settings
from ai_trading_coach.domain.enums import TriggerType
from ai_trading_coach.domain.models import ModelCallTrace, ReviewRunRequest
from ai_trading_coach.llm.gateway import _validate_schema_cached, utc_now
from ai_trading_coach.modules.agent.research_tools import resolve_research_tools
from ai_trading_coach.modules.mcp.mcp_client_manager import MCPClientManager
from ai_trading_coach.orchestrator.langgraph_graph import build_review_graph
from ai_trading_coach.orchestrator.langgraph_nodes import LangGraphNodeRuntime
import ai_trading_coach.orchestrator.langgraph_nodes as langgraph_nodes_module

NODE_ORDER = [
    "parse_log",
    "plan_research",
    "execute_collection",
    "verify_information",
    "build_report_context",
    "generate_report",
    "judge_report",
    "finalize_result",
    "finalize_failure",
]

RUNTIME_METHOD_BY_NODE = {
    "parse_log": "parse_log",
    "plan_research": "plan_research_node",
    "execute_collection": "execute_collection_node",
    "verify_information": "verify_information_node",
    "build_report_context": "build_report_context",
    "generate_report": "generate_report",
    "judge_report": "judge_report",
    "finalize_result": "finalize_result",
    "finalize_failure": "finalize_failure",
}


class Style:
    def __init__(self) -> None:
        self.enabled = sys.stdout.isatty()
        self.reset = "\033[0m" if self.enabled else ""
        self.dim = "\033[2m" if self.enabled else ""
        self.bold = "\033[1m" if self.enabled else ""
        self.blue = "\033[94m" if self.enabled else ""
        self.cyan = "\033[96m" if self.enabled else ""
        self.green = "\033[92m" if self.enabled else ""
        self.yellow = "\033[93m" if self.enabled else ""
        self.red = "\033[91m" if self.enabled else ""
        self.magenta = "\033[95m" if self.enabled else ""


STYLE = Style()


@dataclass
class LLMCapture:
    capture_id: str
    node_name: str | None
    kind: str
    purpose: str | None
    prompt_version: str | None
    schema_name: str | None
    started_at: str
    ended_at: str | None = None
    latency_ms: int | None = None
    input_messages: list[dict[str, Any]] = field(default_factory=list)
    input_summary: str | None = None
    raw_response: Any = None
    raw_text: str | None = None
    parsed_response: Any = None
    trace: dict[str, Any] | None = None
    usage: dict[str, Any] | None = None
    error: str | None = None
    extras: dict[str, Any] = field(default_factory=dict)


class DebugCaptureStore:
    def __init__(self, artifact_root: Path) -> None:
        self.artifact_root = artifact_root
        self.current_node: str | None = None
        self.captures: list[LLMCapture] = []
        self.node_capture_counts: dict[str, int] = {}
        self.node_occurrence_counts: dict[str, int] = {}
        self.current_node_occurrence: int = 0

    def enter_node(self, node_name: str) -> None:
        self.current_node = node_name
        self.node_occurrence_counts[node_name] = self.node_occurrence_counts.get(node_name, 0) + 1
        self.current_node_occurrence = self.node_occurrence_counts[node_name]

    def exit_node(self) -> None:
        self.current_node = None
        self.current_node_occurrence = 0

    def add_capture(self, capture: LLMCapture) -> LLMCapture:
        self.captures.append(capture)
        return capture

    def captures_for_node_occurrence(self, node_name: str, occurrence: int) -> list[LLMCapture]:
        node_dir = self.node_dir(node_name, occurrence)
        if not node_dir.exists():
            return []
        prefix = f"{node_name}.occ{occurrence}."
        out: list[LLMCapture] = []
        for item in self.captures:
            if item.capture_id.startswith(prefix):
                out.append(item)
        return out

    def next_capture_id(self, kind: str) -> str:
        node = self.current_node or "outside_node"
        occ = self.current_node_occurrence or 1
        key = f"{node}#{occ}"
        idx = self.node_capture_counts.get(key, 0) + 1
        self.node_capture_counts[key] = idx
        return f"{node}.occ{occ}.{kind}.{idx:02d}"

    def node_dir(self, node_name: str, occurrence: int) -> Path:
        return self.artifact_root / f"{occurrence:02d}_{node_name}"


def _banner(text: str, *, color: str = "blue") -> None:
    shade = getattr(STYLE, color, "")
    bar = "═" * max(18, len(text) + 4)
    print(f"\n{shade}{STYLE.bold}{bar}\n  {text}\n{bar}{STYLE.reset}")


def _section(text: str, *, color: str = "cyan") -> None:
    shade = getattr(STYLE, color, "")
    print(f"\n{shade}{STYLE.bold}▶ {text}{STYLE.reset}")


def _kv(key: str, value: Any, *, color: str = "dim") -> None:
    shade = getattr(STYLE, color, "")
    print(f"  {STYLE.bold}{key}:{STYLE.reset} {shade}{value}{STYLE.reset}")


def _safe_jsonable(value: Any, *, max_depth: int = 8) -> Any:
    if max_depth <= 0:
        return "<max_depth_reached>"
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "model_dump"):
        try:
            return _safe_jsonable(value.model_dump(mode="json"), max_depth=max_depth - 1)
        except Exception:
            return repr(value)
    if isinstance(value, dict):
        return {str(k): _safe_jsonable(v, max_depth=max_depth - 1) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_safe_jsonable(v, max_depth=max_depth - 1) for v in value]
    if hasattr(value, "content") or hasattr(value, "additional_kwargs") or hasattr(value, "response_metadata"):
        return _serialize_message_like(value, max_depth=max_depth - 1)
    if hasattr(value, "__dict__"):
        return _safe_jsonable(vars(value), max_depth=max_depth - 1)
    return repr(value)


def _pretty_json(value: Any) -> str:
    return json.dumps(_safe_jsonable(value), ensure_ascii=False, indent=2)


def _truncate(text: str, limit: int = 160) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + " ..."


def _extract_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    content = getattr(value, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
                else:
                    parts.append(json.dumps(_safe_jsonable(item), ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    if hasattr(value, "text") and isinstance(getattr(value, "text"), str):
        return getattr(value, "text")
    return str(value)


def _serialize_message_like(value: Any, *, max_depth: int = 6) -> dict[str, Any]:
    role = getattr(value, "type", None) or getattr(value, "role", None) or value.__class__.__name__
    data = {
        "type": value.__class__.__name__,
        "role": role,
        "content": _safe_jsonable(getattr(value, "content", None), max_depth=max_depth - 1),
    }
    for field in ("name", "tool_calls", "invalid_tool_calls", "additional_kwargs", "response_metadata", "usage_metadata"):
        if hasattr(value, field):
            raw = getattr(value, field)
            if raw not in (None, {}, []):
                data[field] = _safe_jsonable(raw, max_depth=max_depth - 1)
    return data


def _extract_usage(response: Any) -> dict[str, Any] | None:
    usage = getattr(response, "usage_metadata", None)
    if isinstance(usage, dict) and usage:
        return _safe_jsonable(usage)
    meta = getattr(response, "response_metadata", None)
    if isinstance(meta, dict):
        inner = meta.get("usage_metadata") or meta.get("token_usage")
        if inner:
            return _safe_jsonable(inner)
    return None


def _build_request(args: argparse.Namespace, *, log_text: str) -> ReviewRunRequest:
    return ReviewRunRequest(
        run_id=args.run_id or f"debug_{args.user_id}_{args.run_date}",
        user_id=args.user_id,
        run_date=date.fromisoformat(args.run_date),
        trigger_type=TriggerType.MANUAL,
        raw_log_text=log_text,
        options={"dry_run": args.dry_run, "debug_mode": args.debug_mode},
    )


def _build_initial_state(request: ReviewRunRequest) -> dict[str, Any]:
    return {
        "request": request,
        "agent_messages": [],
        "rewrite_count": 0,
        "model_calls": [],
        "tool_calls": [],
        "run_started_at": datetime.now(timezone.utc),
        "accumulated_evidence_items": [],
        "accumulated_tool_failures": 0,
        "research_retry_count": 0,
    }


def _node_requires_tools(node: str) -> bool:
    if node == "all":
        return True
    try:
        return NODE_ORDER.index(node) >= NODE_ORDER.index("execute_collection")
    except ValueError:
        return False


def _validate_environment(settings: Settings, *, target_node: str) -> None:
    settings.validate_llm_or_raise()
    if not _node_requires_tools(target_node):
        return

    manager = MCPClientManager(settings=settings)
    registrations = resolve_research_tools(settings=settings, mcp_manager=manager)
    enabled = [item for item in registrations if item.available]
    if not enabled:
        available_msg = (
            "No research tools are configured. Configure MCP_SERVERS and/or web tool keys "
            "(BRAVE_API_KEY, FIRECRAWL_API_KEY, AGENT_BROWSER_ENDPOINT)."
        )
        raise SystemExit(available_msg)

    _section("Environment")
    _kv("llm_provider", settings.llm_provider())
    _kv("llm_model", settings.selected_llm_model())
    _kv("enabled_tools", ", ".join(sorted({item.agent_name for item in enabled})))
    skipped = [item for item in registrations if not item.available]
    if skipped:
        preview = "; ".join(f"{item.agent_name} ({item.reason})" for item in skipped[:4])
        _kv("skipped_tools", preview)


def _iter_node_updates(compiled_graph: Any, initial_state: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    try:
        stream_iter = compiled_graph.stream(initial_state, stream_mode="updates")
    except TypeError:
        stream_iter = compiled_graph.stream(initial_state)

    for event in stream_iter:
        if not isinstance(event, dict):
            yield "__raw_event__", {"event": repr(event)}
            continue

        matched = False
        for key, value in event.items():
            if key in NODE_ORDER:
                matched = True
                if isinstance(value, dict):
                    yield key, value
                else:
                    yield key, {"value": value}

        if matched:
            continue

        state_keys = {
            "request",
            "parse_result",
            "research_output",
            "report_context",
            "report_draft",
            "judge_verdict",
            "final_result",
            "rewrite_count",
            "tool_calls",
            "model_calls",
            "evidence_packet",
        }
        if state_keys & set(event.keys()):
            yield "__state_snapshot__", event
        else:
            yield "__raw_event__", {"event": event}


def _merge_state(state: dict[str, Any], delta: dict[str, Any]) -> None:
    state.update(delta)


def _state_summary(state: dict[str, Any]) -> dict[str, Any]:
    parse_result = state.get("parse_result")
    evidence_packet = state.get("evidence_packet")
    research_output = state.get("research_output")
    feedback = state.get("judgement_feedback") or []
    verdict = state.get("judge_verdict")
    final_result = state.get("final_result")

    judgement_count = len(parse_result.all_judgements()) if parse_result is not None else 0
    evidence_count = 0
    source_count = 0
    if evidence_packet is not None:
        evidence_count = sum(
            len(getattr(evidence_packet, field))
            for field in (
                "price_evidence",
                "news_evidence",
                "filing_evidence",
                "sentiment_evidence",
                "market_regime_evidence",
                "discussion_evidence",
                "analog_evidence",
                "macro_evidence",
            )
        )
        source_count = len(getattr(evidence_packet, "source_registry", []))

    research_count = len(getattr(research_output, "judgement_evidence", [])) if research_output is not None else 0
    return {
        "judgements": judgement_count,
        "evidence_items": evidence_count,
        "sources": source_count,
        "research_links": research_count,
        "feedback_items": len(feedback),
        "rewrite_count": state.get("rewrite_count", 0),
        "model_calls": len(state.get("model_calls", [])),
        "tool_calls": len(state.get("tool_calls", [])),
        "judge_passed": None if verdict is None else getattr(verdict, "passed", None),
        "final_status": None if final_result is None else getattr(final_result, "status", None),
    }


def _node_artifact_dir(store: DebugCaptureStore, node_name: str) -> Path:
    occ = store.node_occurrence_counts.get(node_name, 1)
    path = store.node_dir(node_name, occ)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, data: Any) -> None:
    _write_text(path, _pretty_json(data) + "\n")


def _capture_to_markdown(capture: LLMCapture) -> str:
    usage = capture.usage or {}
    trace = capture.trace or {}
    parts = [
        f"# {capture.capture_id}",
        "",
        "## Metadata",
        "",
        f"- node: `{capture.node_name}`",
        f"- kind: `{capture.kind}`",
        f"- purpose: `{capture.purpose}`",
        f"- prompt_version: `{capture.prompt_version}`",
        f"- schema: `{capture.schema_name}`",
        f"- started_at: `{capture.started_at}`",
        f"- ended_at: `{capture.ended_at}`",
        f"- latency_ms: `{capture.latency_ms}`",
        f"- error: `{capture.error}`",
    ]
    if usage:
        parts.extend([
            "",
            "## Usage",
            "",
            "```json",
            _pretty_json(usage),
            "```",
        ])
    if trace:
        parts.extend([
            "",
            "## Trace",
            "",
            "```json",
            _pretty_json(trace),
            "```",
        ])
    if capture.input_messages:
        parts.extend(["", "## Input Messages", ""])
        for idx, msg in enumerate(capture.input_messages, start=1):
            role = msg.get("role") or msg.get("type") or "message"
            content = msg.get("content", "")
            parts.extend([
                f"### Message {idx} ({role})",
                "",
                "```json" if not isinstance(content, str) else "```text",
                _pretty_json(content) if not isinstance(content, str) else str(content),
                "```",
            ])
            extras = {k: v for k, v in msg.items() if k not in {"role", "type", "content"}}
            if extras:
                parts.extend([
                    "",
                    "```json",
                    _pretty_json(extras),
                    "```",
                ])
    if capture.raw_text:
        parts.extend([
            "",
            "## Raw Model Text",
            "",
            "```text",
            capture.raw_text,
            "```",
        ])
    if capture.raw_response is not None:
        parts.extend([
            "",
            "## Raw Response Object",
            "",
            "```json",
            _pretty_json(capture.raw_response),
            "```",
        ])
    if capture.parsed_response is not None:
        parts.extend([
            "",
            "## Parsed / Validated Output",
            "",
            "```json",
            _pretty_json(capture.parsed_response),
            "```",
        ])
    if capture.extras:
        parts.extend([
            "",
            "## Extras",
            "",
            "```json",
            _pretty_json(capture.extras),
            "```",
        ])
    return "\n".join(parts).strip() + "\n"


def _persist_node_artifacts(store: DebugCaptureStore, node_name: str, delta: dict[str, Any], state: dict[str, Any]) -> Path:
    occurrence = store.node_occurrence_counts.get(node_name, 1)
    node_dir = store.node_dir(node_name, occurrence)
    node_dir.mkdir(parents=True, exist_ok=True)
    _write_json(node_dir / "node_delta.json", delta)
    _write_json(node_dir / "state_summary.json", _state_summary(state))
    _write_json(node_dir / "node_state_snapshot.json", {k: v for k, v in state.items() if k != "request"})

    captures = store.captures_for_node_occurrence(node_name, occurrence)
    llm_dir = node_dir / "llm_calls"
    if captures:
        llm_dir.mkdir(parents=True, exist_ok=True)
    for idx, capture in enumerate(captures, start=1):
        stem = f"{idx:02d}_{capture.kind}_{capture.purpose or 'unknown'}"
        _write_json(llm_dir / f"{stem}.json", capture.__dict__)
        _write_text(llm_dir / f"{stem}.md", _capture_to_markdown(capture))

    if node_name == "execute_collection":
        if state.get("agent_messages") is not None:
            _write_json(node_dir / "agent_messages.json", state.get("agent_messages", []))
        if state.get("react_steps") is not None:
            _write_json(node_dir / "react_steps.json", state.get("react_steps", []))
        if state.get("tool_calls") is not None:
            _write_json(node_dir / "tool_calls.json", state.get("tool_calls", []))
        if state.get("research_output") is not None:
            _write_json(node_dir / "research_output.json", state.get("research_output"))
    return node_dir


def _print_delta(node_name: str, delta: dict[str, Any], state: dict[str, Any]) -> None:
    _section(f"Node: {node_name}", color="green" if node_name.startswith("finalize") else "cyan")

    if node_name == "parse_log":
        parse_result = delta.get("parse_result")
        if parse_result is not None:
            _kv("trade_actions", len(parse_result.trade_actions))
            _kv("all_judgements", len(parse_result.all_judgements()))
            _kv("reflection_summary", len(parse_result.reflection_summary))
            ids = [j.judgement_id for j in parse_result.all_judgements()]
            _kv("judgement_ids", ", ".join(ids) if ids else "-")

    elif node_name == "plan_research":
        _kv("analysis_framework", delta.get("analysis_framework", ""))
        _kv("analysis_directions", len(delta.get("analysis_directions", [])))
        _kv("info_requirements", len(delta.get("info_requirements", [])))

    elif node_name == "execute_collection":
        research_output = delta.get("research_output")
        evidence_packet = delta.get("evidence_packet")
        _kv("agent_messages_total", len(delta.get("agent_messages", [])))
        _kv("tool_calls_total", len(delta.get("tool_calls", [])))
        _kv("react_steps_total", len(delta.get("react_steps", [])))
        if research_output is not None:
            _kv("judgement_evidence", len(research_output.judgement_evidence))
            _kv("stop_reason", research_output.stop_reason)
        if evidence_packet is not None:
            _kv("source_registry", len(evidence_packet.source_registry))
        tool_calls = delta.get("tool_calls", [])
        if tool_calls:
            latest = tool_calls[-min(3, len(tool_calls)):]
            preview = "; ".join(
                f"{call.get('tool_name', '?')}={'ok' if call.get('success', False) else 'fail'}"
                for call in latest
            )
            _kv("recent_tools", preview)

    elif node_name == "verify_information":
        _kv("is_sufficient", delta.get("is_sufficient"))
        _kv("research_stop_reason", delta.get("research_stop_reason"))
        _kv("research_retry_count", delta.get("research_retry_count"))
        if delta.get("insufficiency_reason"):
            _kv("insufficiency_reason", delta.get("insufficiency_reason"), color="yellow")

    elif node_name == "build_report_context":
        report_context = delta.get("report_context", {})
        _kv("judgement_bundles", len(report_context.get("judgement_bundles", [])))
        _kv("source_index", len(report_context.get("source_index", [])))

    elif node_name == "generate_report":
        report_draft = delta.get("report_draft", "")
        feedback = delta.get("judgement_feedback", [])
        _kv("report_chars", len(report_draft))
        _kv("feedback_items", len(feedback))
        if report_draft:
            first_line = next((line.strip() for line in report_draft.splitlines() if line.strip()), "")
            _kv("report_preview", _truncate(first_line), color="dim")

    elif node_name == "judge_report":
        verdict = delta.get("judge_verdict")
        _kv("passed", getattr(verdict, "passed", None))
        _kv("rewrite_count", delta.get("rewrite_count"))
        reasons = getattr(verdict, "reasons", None)
        if reasons:
            _kv("reasons", "; ".join(reasons[:3]))
        rewrite_instruction = delta.get("rewrite_instruction")
        if rewrite_instruction:
            _kv("rewrite_instruction", _truncate(rewrite_instruction, 220), color="yellow")

    elif node_name == "finalize_result":
        final_result = delta.get("final_result")
        _kv("status", getattr(final_result, "status", None), color="green")
        report = getattr(final_result, "report", None)
        if report is not None:
            _kv("report_title", report.title)
            _kv("report_chars", len(report.markdown_body))

    elif node_name == "finalize_failure":
        final_result = delta.get("final_result")
        _kv("status", getattr(final_result, "status", None), color="red")
        errors = getattr(final_result, "errors", [])
        if errors:
            _kv("error", errors[0].message, color="red")

    else:
        _kv("delta_keys", ", ".join(delta.keys()) if delta else "-")

    summary = _state_summary(state)
    _kv("state_summary", summary)


def _print_full_json(node_name: str, delta: dict[str, Any]) -> None:
    _kv(f"{node_name}.delta", _pretty_json(delta))


def _print_llm_captures(
    store: DebugCaptureStore,
    node_name: str,
    *,
    artifact_dir: Path,
    print_full: bool,
) -> None:
    occurrence = store.node_occurrence_counts.get(node_name, 1)
    captures = store.captures_for_node_occurrence(node_name, occurrence)
    if not captures:
        _kv("llm_calls", "none")
        return

    _section(f"LLM Captures for {node_name}", color="magenta")
    _kv("count", len(captures))
    _kv("artifact_dir", artifact_dir / "llm_calls")

    for idx, capture in enumerate(captures, start=1):
        _kv(f"call_{idx}", f"kind={capture.kind}; purpose={capture.purpose}; latency_ms={capture.latency_ms}; error={capture.error}")
        if capture.schema_name:
            _kv("schema", capture.schema_name)
        if capture.input_summary:
            _kv("input_summary", capture.input_summary)
        if capture.usage:
            _kv("usage", capture.usage)
        preview = capture.raw_text or _pretty_json(capture.parsed_response) if capture.parsed_response is not None else ""
        if preview:
            _kv("output_preview", _truncate(preview, 220))
        if print_full:
            md = _capture_to_markdown(capture)
            print(textwrap.indent(md.rstrip(), prefix="    "))


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Debug the real LangGraph review pipeline node-by-node.")
    parser.add_argument("--node", choices=[*NODE_ORDER, "all"], default="all", help="Stop after this node executes. Use 'all' to run the full graph.")
    parser.add_argument("--occurrence", type=int, default=1, help="Stop after the Nth occurrence of the target node.")
    parser.add_argument("--user-id", default="debug_user")
    parser.add_argument("--run-id", default="", help="Optional explicit run_id.")
    parser.add_argument("--run-date", default=date.today().isoformat())
    parser.add_argument("--log-file", default="examples/logs/daily_log_sample.md")
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=True, help="Default true. Keep long-term memory/report writes disabled unless explicitly turned off.")
    parser.add_argument("--debug-mode", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--print-json", action="store_true", help="Print the full node delta as JSON-like output.")
    parser.add_argument("--print-request", action="store_true", help="Print the request payload before execution.")
    parser.add_argument("--print-llm-full", action="store_true", help="Print full LLM inputs/outputs inline in the terminal. Full captures are always saved to disk.")
    parser.add_argument("--artifact-root", default="debug_artifacts", help="Directory (relative to repo root) for saved debug artifacts.")
    parser.add_argument("--traceback", action="store_true", help="Print full Python traceback on failure.")
    return parser


def _wrap_runtime_node_methods(runtime: LangGraphNodeRuntime, store: DebugCaptureStore) -> None:
    for node_name, method_name in RUNTIME_METHOD_BY_NODE.items():
        original = getattr(runtime, method_name)

        def _wrapped(self: LangGraphNodeRuntime, state: dict[str, Any], *, _orig: Callable[..., Any] = original, _node_name: str = node_name):
            store.enter_node(_node_name)
            try:
                return _orig(state)
            finally:
                store.exit_node()

        setattr(runtime, method_name, MethodType(_wrapped, runtime))


def _patch_gateway_for_debug(gateway: Any, store: DebugCaptureStore) -> None:
    def invoke_structured_debug(
        self: Any,
        *,
        schema: type[Any],
        messages: list[dict[str, str]],
        purpose: Any,
        prompt_version: str,
        input_summary: str,
        output_summary_builder: Callable[[Any], str] | None = None,
    ) -> tuple[Any, ModelCallTrace]:
        _validate_schema_cached(schema)
        started_at_dt = utc_now()
        started_perf = perf_counter()
        capture = LLMCapture(
            capture_id=store.next_capture_id("structured"),
            node_name=store.current_node,
            kind="structured",
            purpose=getattr(purpose, "value", str(purpose)),
            prompt_version=prompt_version,
            schema_name=schema.__name__,
            started_at=started_at_dt.isoformat(),
            input_messages=_safe_jsonable(messages),
            input_summary=input_summary,
        )
        raw_response_obj: Any = None
        parsed_result: Any = None
        include_raw_used = False
        try:
            with_structured_sig = inspect.signature(self.model.with_structured_output)
            include_raw_used = "include_raw" in with_structured_sig.parameters
        except Exception:
            include_raw_used = False

        try:
            if include_raw_used:
                structured_model = self.model.with_structured_output(schema, include_raw=True)
            else:
                structured_model = self.model.with_structured_output(schema)
            raw = self._invoke_structured_model_with_warning_filter(structured_model, messages)
            raw_response_obj = raw
            if include_raw_used and isinstance(raw, dict) and {"raw", "parsed"}.issubset(raw.keys()):
                raw_response_obj = raw.get("raw")
                parsed_candidate = raw.get("parsed")
                parsing_error = raw.get("parsing_error")
                if parsing_error is not None:
                    raise RuntimeError(f"LangChain parsing_error: {parsing_error}")
                parsed_result = parsed_candidate if isinstance(parsed_candidate, schema) else schema.model_validate(parsed_candidate)
                capture.extras["include_raw_used"] = True
            else:
                parsed_result = raw if isinstance(raw, schema) else schema.model_validate(raw)
                capture.extras["include_raw_used"] = False

            ended_at_dt = utc_now()
            latency = int((perf_counter() - started_perf) * 1000)
            output_summary = output_summary_builder(parsed_result) if output_summary_builder else schema.__name__
            trace = self._build_model_call_trace(
                purpose=purpose,
                started_at=started_at_dt,
                ended_at=ended_at_dt,
                prompt_version=prompt_version,
                input_summary=input_summary,
                output_summary=output_summary,
                latency_ms=latency,
            )
            capture.ended_at = ended_at_dt.isoformat()
            capture.latency_ms = latency
            capture.raw_response = _safe_jsonable(raw_response_obj)
            capture.raw_text = _extract_text(raw_response_obj) if raw_response_obj is not None else None
            capture.parsed_response = _safe_jsonable(parsed_result)
            capture.trace = trace.model_dump(mode="json")
            capture.usage = _extract_usage(raw_response_obj)
            store.add_capture(capture)
            return parsed_result, trace
        except Exception as exc:  # noqa: BLE001
            ended_at_dt = utc_now()
            latency = int((perf_counter() - started_perf) * 1000)
            _ = self._build_model_call_trace(
                purpose=purpose,
                started_at=started_at_dt,
                ended_at=ended_at_dt,
                prompt_version=prompt_version,
                input_summary=input_summary,
                output_summary=f"error:{exc.__class__.__name__}",
                latency_ms=latency,
                error_message=str(exc),
            )
            capture.ended_at = ended_at_dt.isoformat()
            capture.latency_ms = latency
            capture.error = f"{exc.__class__.__name__}: {exc}"
            if raw_response_obj is not None:
                capture.raw_response = _safe_jsonable(raw_response_obj)
                capture.raw_text = _extract_text(raw_response_obj)
                capture.usage = _extract_usage(raw_response_obj)
            store.add_capture(capture)
            raise RuntimeError(
                f"Structured output failed for schema={schema.__name__}, "
                f"purpose={purpose.value}, prompt_version={prompt_version}"
            ) from exc

    def invoke_text_debug(
        self: Any,
        *,
        messages: list[dict[str, str]],
        purpose: Any,
        prompt_version: str,
        input_summary: str,
    ) -> tuple[str, ModelCallTrace]:
        started_at_dt = utc_now()
        started_perf = perf_counter()
        capture = LLMCapture(
            capture_id=store.next_capture_id("text"),
            node_name=store.current_node,
            kind="text",
            purpose=getattr(purpose, "value", str(purpose)),
            prompt_version=prompt_version,
            schema_name=None,
            started_at=started_at_dt.isoformat(),
            input_messages=_safe_jsonable(messages),
            input_summary=input_summary,
        )
        response_obj: Any = None
        try:
            response_obj = self.model.invoke(messages)
            ended_at_dt = utc_now()
            latency = int((perf_counter() - started_perf) * 1000)
            content = _extract_text(response_obj)
            trace = self._build_model_call_trace(
                purpose=purpose,
                started_at=started_at_dt,
                ended_at=ended_at_dt,
                prompt_version=prompt_version,
                input_summary=input_summary,
                output_summary=f"chars={len(content)}",
                latency_ms=latency,
            )
            capture.ended_at = ended_at_dt.isoformat()
            capture.latency_ms = latency
            capture.raw_response = _safe_jsonable(response_obj)
            capture.raw_text = content
            capture.trace = trace.model_dump(mode="json")
            capture.usage = _extract_usage(response_obj)
            store.add_capture(capture)
            return content, trace
        except Exception as exc:  # noqa: BLE001
            ended_at_dt = utc_now()
            latency = int((perf_counter() - started_perf) * 1000)
            capture.ended_at = ended_at_dt.isoformat()
            capture.latency_ms = latency
            capture.error = f"{exc.__class__.__name__}: {exc}"
            if response_obj is not None:
                capture.raw_response = _safe_jsonable(response_obj)
                capture.raw_text = _extract_text(response_obj)
                capture.usage = _extract_usage(response_obj)
            store.add_capture(capture)
            raise RuntimeError(
                f"Text output failed for purpose={purpose.value}, prompt_version={prompt_version}"
            ) from exc

    gateway.invoke_structured = MethodType(invoke_structured_debug, gateway)
    gateway.invoke_text = MethodType(invoke_text_debug, gateway)


def _patch_create_agent_for_debug(store: DebugCaptureStore) -> Callable[..., Any]:
    original_create_agent = langgraph_nodes_module.create_agent

    def create_agent_debug(*args: Any, **kwargs: Any) -> Any:
        agent = original_create_agent(*args, **kwargs)

        class AgentProxy:
            def __init__(self, inner: Any) -> None:
                self._inner = inner

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner, name)

            def invoke(self, inputs: Any, *invoke_args: Any, **invoke_kwargs: Any) -> Any:
                started_at_dt = utc_now()
                started_perf = perf_counter()
                capture = LLMCapture(
                    capture_id=store.next_capture_id("agent"),
                    node_name=store.current_node,
                    kind="agent",
                    purpose="research_agent",
                    prompt_version="research_agent.runtime",
                    schema_name="ResearchAgentFinalContract",
                    started_at=started_at_dt.isoformat(),
                    input_messages=[{"role": "agent.invoke", "content": _safe_jsonable(inputs)}],
                    input_summary="real create_agent invoke",
                )
                try:
                    result = self._inner.invoke(inputs, *invoke_args, **invoke_kwargs)
                    ended_at_dt = utc_now()
                    capture.ended_at = ended_at_dt.isoformat()
                    capture.latency_ms = int((perf_counter() - started_perf) * 1000)
                    capture.raw_response = _safe_jsonable(result)
                    messages = result.get("messages", []) if isinstance(result, dict) else []
                    text_blocks = [_extract_text(msg) for msg in messages]
                    capture.raw_text = "\n\n-----\n\n".join(block for block in text_blocks if block).strip() or None
                    if isinstance(result, dict) and "structured_response" in result:
                        capture.parsed_response = _safe_jsonable(result.get("structured_response"))
                    capture.extras = {
                        "result_keys": sorted(result.keys()) if isinstance(result, dict) else None,
                        "message_count": len(messages),
                    }
                    store.add_capture(capture)
                    return result
                except Exception as exc:  # noqa: BLE001
                    ended_at_dt = utc_now()
                    capture.ended_at = ended_at_dt.isoformat()
                    capture.latency_ms = int((perf_counter() - started_perf) * 1000)
                    capture.error = f"{exc.__class__.__name__}: {exc}"
                    store.add_capture(capture)
                    raise

        return AgentProxy(agent)

    langgraph_nodes_module.create_agent = create_agent_debug
    return original_create_agent


def _build_runtime_and_graph(settings: Settings, store: DebugCaptureStore) -> Any:
    modules = build_orchestrator_modules(settings=settings)
    _patch_gateway_for_debug(modules.llm_gateway, store)
    runtime = LangGraphNodeRuntime(
        parser_agent=modules.parser_agent,
        reporter_agent=modules.reporter_agent,
        report_judge=modules.report_judge,
        context_builder=modules.context_builder,
        mcp_manager=modules.mcp_manager,
        chat_model=modules.llm_gateway.model,
        settings=settings,
        long_term_store=modules.long_term_store,
        prompt_manager=modules.prompt_manager,
    )
    _wrap_runtime_node_methods(runtime, store)
    compiled_graph = build_review_graph(runtime)
    return compiled_graph


def main() -> int:
    args = _build_arg_parser().parse_args()
    log_path = Path(args.log_file)
    if not log_path.is_absolute():
        log_path = ROOT / log_path
    if not log_path.exists():
        raise SystemExit(f"Log file not found: {log_path}")

    artifact_root = Path(args.artifact_root)
    if not artifact_root.is_absolute():
        artifact_root = ROOT / artifact_root
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    artifact_root = artifact_root / f"{run_stamp}_{args.node}"
    artifact_root.mkdir(parents=True, exist_ok=True)

    settings = get_settings()
    store = DebugCaptureStore(artifact_root=artifact_root)
    _banner("Real LangGraph Node Debugger")
    _validate_environment(settings, target_node=args.node)

    log_text = log_path.read_text(encoding="utf-8")
    request = _build_request(args, log_text=log_text)
    initial_state = _build_initial_state(request)

    if args.print_request:
        _section("Request")
        _kv("request", _pretty_json(request))

    _section("Run")
    _kv("target_node", args.node)
    _kv("occurrence", args.occurrence)
    _kv("dry_run", args.dry_run)
    _kv("log_file", str(log_path))
    _kv("log_chars", len(log_text))
    _kv("artifact_root", artifact_root)

    original_create_agent = _patch_create_agent_for_debug(store)
    compiled_graph = _build_runtime_and_graph(settings, store)

    state: dict[str, Any] = dict(initial_state)
    hit_count = 0
    seen_nodes: list[str] = []

    try:
        for node_name, delta in _iter_node_updates(compiled_graph, initial_state):
            if node_name.startswith("__"):
                _section(f"Aux stream event: {node_name}", color="magenta")
                _kv("payload", _pretty_json(delta))
                continue

            _merge_state(state, delta)
            seen_nodes.append(node_name)
            _print_delta(node_name, delta, state)
            if args.print_json:
                _print_full_json(node_name, delta)

            node_dir = _persist_node_artifacts(store, node_name, delta, state)
            _print_llm_captures(store, node_name, artifact_dir=node_dir, print_full=args.print_llm_full)
            _kv("node_artifacts", node_dir)

            if args.node != "all" and node_name == args.node:
                hit_count += 1
                if hit_count >= args.occurrence:
                    _section("Stopped at requested node", color="yellow")
                    _kv("node", node_name)
                    _kv("occurrence", hit_count)
                    _kv("artifact_root", artifact_root)
                    return 0

    except Exception as exc:  # noqa: BLE001
        _banner("Execution Failed", color="red")
        _kv("exception_type", type(exc).__name__, color="red")
        _kv("message", str(exc), color="red")
        if seen_nodes:
            _kv("last_completed_node", seen_nodes[-1], color="yellow")
        _kv("artifact_root", artifact_root, color="yellow")
        _kv("state_summary", _state_summary(state), color="yellow")
        if args.traceback:
            print()
            traceback.print_exc()
        return 1
    finally:
        langgraph_nodes_module.create_agent = original_create_agent

    _banner("Run Complete", color="green")
    _kv("visited_nodes", " -> ".join(seen_nodes) if seen_nodes else "-")
    _kv("artifact_root", artifact_root)
    _kv("final_state_summary", _state_summary(state))
    if args.node != "all":
        _kv("note", f"Target node '{args.node}' was not reached.", color="yellow")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
