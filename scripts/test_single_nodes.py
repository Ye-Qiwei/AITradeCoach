import asyncio
import json
from datetime import datetime, timezone
from rich.console import Console
from rich.panel import Panel
from rich.syntax import Syntax

# 引入项目配置与状态结构
from ai_trading_coach.config import Settings
from ai_trading_coach.domain.models import ReviewRunRequest
from ai_trading_coach.orchestrator.langgraph_state import OrchestratorGraphState

# 假设你的工厂方法或依赖注入函数名为 build_daily_nodes 
# 如果 factory.py 中没有封装整个 nodes 的构建，请自行替换为对应的初始化代码
# 例如：from ai_trading_coach.app.factory import build_daily_nodes 

console = Console()

def print_node_result(node_name: str, result: dict):
    """使用 rich 美化打印 node 的输出"""
    json_str = json.dumps(result, default=str, indent=2, ensure_ascii=False)
    console.print(Panel(Syntax(json_str, "json", theme="monokai", word_wrap=True), 
                        title=f"[bold green]{node_name} Output[/bold green]", 
                        border_style="green"))

async def test_all_nodes_independently():
    console.print(Panel.fit("🚀 Starting LangGraph Nodes Independent Test", style="bold blue"))
    
    # 1. 初始化核心依赖 (复用真实流程)
    settings = Settings()
    
    # 【注意】这里请替换为你项目中真实获取 Nodes 实例的方法。
    # 如果你的系统没有提供统一的方法，可以在这里手动实例化对应的 Agent 和 Nodes 类：
    # 例如: nodes = DailyGraphNodes(parser_agent=..., reporter_agent=...)
    # nodes = build_daily_nodes(settings) 
    
    # 为了演示，假设我们已经拿到了 nodes 实例
    nodes = None # TODO: 替换为实际的 nodes 实例
    if not nodes:
        console.print("[yellow]请在代码中补充实际的 `nodes` 实例获取方式。[/yellow]")
        return

    # 2. 构建符合 Schema 的最小必要虚拟输入
    # ReviewRunRequest 是图的起点
    request = ReviewRunRequest(
        run_id="test_debug_run_001",
        user_id="test_user",
        run_date=datetime.now(timezone.utc),
        raw_log_text="今天大盘回调，我趁机买入了少量的NVDA，感觉AI长期逻辑不变。另外清仓了TSLA，因为预期财报可能不及预期。"
    )
    
    # 按照 OrchestratorGraphState 的 TypedDict 定义初始化 state
    state: OrchestratorGraphState = {"request": request}
    
    # ---------------------------------------------------------
    # Node 1: Parse Node (认知提取与评估)
    # ---------------------------------------------------------
    console.print("\n[bold cyan]▶ Running: Parse Node[/bold cyan]")
    try:
        # 注意：如果你的 node 是同步方法，请去掉 await
        parse_update = await nodes.parse_node(state) 
        state.update(parse_update)
        print_node_result("Parse Node", {"parse_result": parse_update.get("parse_result")})
    except Exception as e:
        console.print(f"[red]Parse Node Failed: {e}[/red]")
        return # 阻断后续测试，因为后续依赖此状态

    # ---------------------------------------------------------
    # Node 2: Evidence / Research Node (证据规划与外部信息获取)
    # ---------------------------------------------------------
    console.print("\n[bold cyan]▶ Running: Research Node[/bold cyan]")
    try:
        # 如果有专门的 evidence packet 构建节点，先运行它
        if hasattr(nodes, 'build_evidence_packet_node'):
            ev_update = await nodes.build_evidence_packet_node(state)
            state.update(ev_update)
            
        research_update = await nodes.research_node(state)
        state.update(research_update)
        print_node_result("Research Node", {"research_output": research_update.get("research_output")})
    except Exception as e:
        console.print(f"[red]Research Node Failed: {e}[/red]")
        return

    # ---------------------------------------------------------
    # Node 3: Report Node (报告生成)
    # ---------------------------------------------------------
    console.print("\n[bold cyan]▶ Running: Report Node[/bold cyan]")
    try:
        report_update = await nodes.report_node(state)
        state.update(report_update)
        # 报告通常是 markdown 文本，单独打印
        console.print(Panel(report_update.get("report_draft", "No draft generated"), 
                            title="[bold green]Report Draft[/bold green]", 
                            border_style="green"))
    except Exception as e:
        console.print(f"[red]Report Node Failed: {e}[/red]")
        return

    # ---------------------------------------------------------
    # Node 4: Judge Node (报告校验与反馈)
    # ---------------------------------------------------------
    console.print("\n[bold cyan]▶ Running: Judge Node[/bold cyan]")
    try:
        judge_update = await nodes.judge_node(state)
        state.update(judge_update)
        print_node_result("Judge Node", {
            "judge_verdict": judge_update.get("judge_verdict"),
            "judgement_feedback": judge_update.get("judgement_feedback")
        })
    except Exception as e:
        console.print(f"[red]Judge Node Failed: {e}[/red]")

    console.print("\n[bold blue]✅ All nodes execution completed.[/bold blue]")

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(test_all_nodes_independently())