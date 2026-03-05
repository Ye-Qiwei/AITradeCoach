# Prompt Registry (Versioned)

This directory stores production-safe prompt templates.

Rules:
- No module should hardcode a long prompt string in business logic.
- Prompt changes must create a new versioned file and be evaluated offline.
- Every prompt must declare: input contract, output schema, constraints, failure handling.

Current baseline version: `baseline_v1`

Active versions are tracked in `manifest.json`.
