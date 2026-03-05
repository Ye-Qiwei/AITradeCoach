# Prompt: Review Report Generation (v1)

## Input Contract
- `EvaluationResult`
- Position/PnL snapshots
- `EvidencePacket`
- `WindowDecision`

## Output Schema
- `DailyReviewReport`

## Constraints
- Coach tone, evidence first
- Must include strengths and mistakes symmetrically
- Must provide next-step observation checklist

## Failure Handling
- If evidence coverage is weak, include explicit uncertainty section

## Style
- Markdown sections + structured summary
