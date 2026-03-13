# AITradeCoach (MVP)

## Quick start

1. Create venv and install:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -e .
   ```
2. Copy env:
   ```bash
   cp .env.example .env
   ```
3. Fill minimum keys:
   - `LLM_PROVIDER`, `OPENAI_API_KEY` (or Gemini)
   - `MCP_SERVERS` (yfinance)
4. Run health checks:
   ```bash
   python scripts/debug_langgraph_nodes.py --node execute_collection
   ```
5. Run manual app:
   ```bash
   python -m ai_trading_coach.app.run_manual
   ```

## Unified tool system

All agent tools are registered in one pipeline:

- `build_runtime_research_tools(...)` builds every tool from unified `RuntimeToolSpec`
- each tool exposes strict `args_schema` (required fields visible to model)
- backend can be MCP / HTTP / local Python, but exposure to agent is identical

Current MVP tools:
- `get_price_history` (MCP yfinance)
- `search_news` (MCP yfinance)
- `yahoo_japan_fund_history` (local python)
- `brave_search` / `firecrawl_extract` / `playwright_fetch` (web)

No `MCP_TOOL_ALLOWLIST` is used in MVP.
