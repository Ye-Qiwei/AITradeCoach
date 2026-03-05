# Prompt: Log Understanding (v1)

## Input Contract
- `DailyLogRaw`

## Output Schema
- `DailyLogNormalized`

## Constraints
- Preserve raw lines in `trade_narratives`
- Emit field-level parsing issues in `field_errors`
- Do not infer missing facts as confirmed facts

## Failure Handling
- If ticker/price parse fails, keep raw line and emit warning
- If date missing, fallback to run date with warning

## Style
- Structured JSON only
