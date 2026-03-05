"""Protocol interfaces for pluggable modules."""

from .modules import (
    CognitionExtractionEngine,
    CognitionRealityEvaluator,
    DailyLogIntakeCanonicalizer,
    DynamicAnalysisWindowSelector,
    EvidencePlanner,
    LongTermMemoryService,
    MCPToolGateway,
    PromptOpsSelfImprovementEngine,
    ReviewReportGenerator,
    ShortTermContextBuilder,
    SystemOrchestrator,
    TradeLedgerPositionEngine,
)

__all__ = [
    "SystemOrchestrator",
    "DailyLogIntakeCanonicalizer",
    "TradeLedgerPositionEngine",
    "CognitionExtractionEngine",
    "LongTermMemoryService",
    "ShortTermContextBuilder",
    "EvidencePlanner",
    "MCPToolGateway",
    "DynamicAnalysisWindowSelector",
    "CognitionRealityEvaluator",
    "ReviewReportGenerator",
    "PromptOpsSelfImprovementEngine",
]
