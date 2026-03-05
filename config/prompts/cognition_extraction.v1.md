# Prompt: Cognition Extraction (v1)

## Input Contract
- `DailyLogNormalized`

## Output Schema
- `CognitionState`

## Constraints
- Separate `fact_statements` vs `subjective_statements`
- Prefer falsifiable `Hypothesis`
- Keep explicit rules and fuzzy tendencies separate

## Failure Handling
- If confidence low, set `Hypothesis.status=pending` and reduce confidence
- Avoid hallucinating unsupported thesis

## Style
- Structured JSON with concise evidence strings
