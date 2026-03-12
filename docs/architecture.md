# Architecture

Official LangGraph mainline:

`parse_log -> plan_research -> execute_collection -> verify_information -> build_report_context -> generate_report -> judge_report -> finalize_result/finalize_failure`

## Tool surface
- Agent sees **only curated tools**.
- Curated tools route by two paths:
  - local: curated tool -> Python function (`yahoo_japan_fund_history`)
  - external MCP: curated tool -> `MCPClientManager.call_tool`
- Raw MCP discovery metadata is kept for doctor/diagnostics and mapping audit; raw tool names are not injected to the research agent.

## MCP dependency policy
- Yahoo Finance MCP provider is `narumiruna/yfinance-mcp` (e.g., `uvx yfmcp@latest`).
- This is still an MCP server process (not direct SaaS API), but no repo clone/project venv is required for normal usage.
