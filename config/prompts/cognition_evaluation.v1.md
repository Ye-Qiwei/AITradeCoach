# Prompt: Cognition vs Reality Evaluation (v1)

## Input Contract
- `CognitionState`
- `EvidencePacket`
- `WindowDecision`
- Relevant memory and position snapshot

## Output Schema
- `EvaluationResult`

## Constraints
- Three-layer output: fact, interpretation, evaluation
- Distinguish: right but early / right but execution poor / wrong
- No hindsight-only judgement

## Failure Handling
- If evidence conflicts heavily, mark uncertainty and propose follow-up signals

## Style
- JSON only, evidence-linked findings
