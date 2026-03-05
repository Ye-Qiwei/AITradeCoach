# Prompt: PromptOps Self Improvement (v1)

## Input Contract
- Historical quality feedback
- Current run metrics and failure cases

## Output Schema
- `ImprovementProposal` (+ optional candidates)

## Constraints
- Proposal only; no direct production prompt mutation
- Include offline eval plan and success metrics

## Failure Handling
- If no clear issue, output `status=proposed` with low-priority monitoring proposal

## Style
- Structured JSON for proposal lifecycle
