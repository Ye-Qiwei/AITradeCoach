"""Domain enumerations for the coaching system."""

from __future__ import annotations

from enum import Enum


class TriggerType(str, Enum):
    SCHEDULED = "scheduled"
    MANUAL = "manual"
    REPLAY = "replay"


class RunStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    SKIPPED = "skipped"


class SourceType(str, Enum):
    MARKDOWN = "markdown"
    JSON = "json"
    SNIPPET = "snippet"
    UNKNOWN = "unknown"


class AssetType(str, Enum):
    STOCK = "stock"
    ETF = "etf"
    FUND = "fund"
    INDEX = "index"
    OPTION = "option"
    FUTURE = "future"
    OTHER = "other"


class TradeSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class HypothesisType(str, Enum):
    SHORT_CATALYST = "short_catalyst"
    MID_THESIS = "mid_thesis"
    LONG_THESIS = "long_thesis"
    RISK_HEDGE = "risk_hedge"
    EXECUTION_RULE = "execution_rule"
    OTHER = "other"


class HypothesisStatus(str, Enum):
    ACTIVE = "active"
    CONFIRMED = "confirmed"
    WEAKENED = "weakened"
    INVALIDATED = "invalidated"
    PENDING = "pending"


class MemoryType(str, Enum):
    RAW_LOG = "raw_log"
    COGNITIVE_CASE = "cognitive_case"
    USER_PROFILE = "user_profile"
    ACTIVE_THESIS = "active_thesis"
    IMPROVEMENT_NOTE = "improvement_note"


class MemoryStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    INVALIDATED = "invalidated"


class EvidenceType(str, Enum):
    PRICE_PATH = "price_path"
    FILING = "filing"
    NEWS = "news"
    SECTOR_LINKAGE = "sector_linkage"
    SENTIMENT = "sentiment"
    DISCUSSION = "discussion"
    MACRO = "macro"
    ANALOG_HISTORY = "analog_history"


class AnalysisWindowType(str, Enum):
    INTRADAY = "intraday"
    D1 = "1D"
    D5 = "5D"
    D20 = "20D"
    D60 = "60D"
    D120 = "120D"
    D252 = "252D"
    SINCE_ENTRY = "since_entry"
    EVENT_CENTERED = "event_centered_window"
    ANALOG_SEGMENT = "analog_historical_segment"
    MULTI_WINDOW = "multi_window_comparison"


class JudgementType(str, Enum):
    FINAL = "final"
    PRELIMINARY = "preliminary"
    FOLLOW_UP_REQUIRED = "follow_up_required"


class EvaluationCategory(str, Enum):
    CORRECT = "judgement_correct"
    WRONG = "judgement_wrong"
    PARTIAL = "judgement_partial"
    DIRECTION_RIGHT_TIMING_WRONG = "direction_right_timing_wrong"
    AHEAD_OF_MARKET = "ahead_of_market"
    CORRECT_BUT_EXECUTION_POOR = "correct_but_execution_poor"
    CORRECT_BUT_POSITION_INAPPROPRIATE = "correct_but_position_inappropriate"
    EMOTION_AMPLIFIED_ERROR = "emotion_amplified_error"
    NOT_TRADE_EXCELLENT = "not_trade_excellent"
    FOMO_TOO_STRONG = "fomo_too_strong"
    RISK_AWARENESS_INSUFFICIENT = "risk_awareness_insufficient"


class BiasType(str, Enum):
    INFORMATION = "information_bias"
    LOGIC = "logic_bias"
    TIME_SCALE = "time_scale_bias"
    EXECUTION = "execution_bias"
    EMOTION = "emotion_bias"
    EVIDENCE_SELECTION = "evidence_selection_bias"


class ImprovementScope(str, Enum):
    PROMPT = "prompt"
    CONTEXT_POLICY = "context_policy"
    RETRIEVAL = "retrieval"
    WINDOW_SELECTION = "window_selection"
    BIAS_RULE = "bias_rule"
    REPORT_STYLE = "report_style"
    TOOL_SEQUENCE = "tool_sequence"


class ProposalStatus(str, Enum):
    PROPOSED = "proposed"
    OFFLINE_EVALUATING = "offline_evaluating"
    AB_TESTING = "ab_testing"
    PROMOTED = "promoted"
    REJECTED = "rejected"


class ModelCallPurpose(str, Enum):
    LOG_UNDERSTANDING = "log_understanding"
    COGNITION_EXTRACTION = "cognition_extraction"
    EVIDENCE_PLANNING = "evidence_planning"
    WINDOW_SELECTION = "window_selection"
    COGNITION_EVALUATION = "cognition_evaluation"
    REPORT_GENERATION = "report_generation"
    IMPROVEMENT_PROPOSAL = "improvement_proposal"


class ModuleName(str, Enum):
    ORCHESTRATOR = "orchestrator"
    LOG_INTAKE = "daily_log_intake"
    LEDGER_ENGINE = "trade_ledger_engine"
    COGNITION_ENGINE = "cognition_extraction_engine"
    MEMORY_SERVICE = "long_term_memory_service"
    CONTEXT_BUILDER = "short_term_context_builder"
    EVIDENCE_PLANNER = "evidence_planner"
    MCP_GATEWAY = "mcp_tool_gateway"
    WINDOW_SELECTOR = "dynamic_window_selector"
    EVALUATOR = "cognition_reality_evaluator"
    REPORT_GENERATOR = "review_report_generator"
    PROMPTOPS = "promptops_self_improvement_engine"
