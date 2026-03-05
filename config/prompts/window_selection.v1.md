# Prompt: Dynamic Window Selection (v1)

## Input Contract
- `EvidencePlan`
- `CognitionState`
- `TradeLedger`/`PositionSnapshot`
- Event and volatility hints

## Output Schema
- `WindowDecision`

## Constraints
- Must output `selected_windows` and `rejected_windows`
- Include reasoning and confidence per selected window
- Explicitly decide follow-up necessity

## Failure Handling
- If evidence is insufficient, set `judgement_type=follow_up_required`

## Style
- JSON only, deterministic labels
