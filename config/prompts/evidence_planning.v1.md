# Prompt: Evidence Planning (v1)

## Input Contract
- `CognitionState`
- Active thesis memory records
- Relevant history

## Output Schema
- `EvidencePlan`

## Constraints
- Claim-driven planning only
- Each claim maps to concrete evidence types and time ranges
- Include priority and event-centered/analog flags

## Failure Handling
- If claims are vague, generate clarification questions in `planner_notes`

## Style
- JSON only
