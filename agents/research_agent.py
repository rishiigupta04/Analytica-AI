# =============================================================
# agents/research_agent.py — The Research Agent (Phase 3 — updated)
# =============================================================
# WHAT CHANGED FROM PHASE 2:
#   ★ Reads state["research_retry_count"] — retry awareness
#   ★ Reads state["critic_research_feedback"] — targeted improvement
#   ★ Increments research_retry_count in return value
#   ★ On retry: narrows search query based on critic's feedback
#     and refines the summarization prompt
# =============================================================

import time
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage, SystemMessage

from state import MarketingState
from utils.llm import get_llm


def research_agent(state: MarketingState) -> dict:
    """
    Research Agent Node (Phase 3).

    Reads:  state["external_tasks"],
            state["research_retry_count"] ★, state["critic_research_feedback"] ★
    Writes: state["research_output"], state["research_retry_count"] ★
    """
    if not state.get("external_tasks"):
        return {
            "research_output": "No external research required for this query.",
            "research_retry_count": 0,
        }

    # ★ Phase 3: retry awareness
    retry_count = state.get("research_retry_count", 0)
    critic_feedback = state.get("critic_research_feedback", "")

    attempt_label = f"attempt {retry_count + 1}/3" if retry_count > 0 else "first attempt"
    print(f"\n🔍 [RESEARCH AGENT] {attempt_label} — "
          f"{len(state['external_tasks'])} task(s)")

    llm = get_llm(temperature=0)
    search = DuckDuckGoSearchRun()
    all_findings = []

    # ★ Build retry improvement note for the summarization prompt
    retry_note = ""
    if retry_count > 0 and critic_feedback:
        retry_note = (
            f"\n\n⚠️ QUALITY IMPROVEMENT REQUIRED:\n"
            f"Previous research was flagged: {critic_feedback}\n"
            f"This time: find MORE SPECIFIC data — include statistics, "
            f"percentages, named strategies, and year references."
        )

    for task in state["external_tasks"]:
        print(f"   🌐 Researching: {task['task'][:60]}...")

        # Step 1: Generate optimized search query
        # ★ On retry, prompt for more specific/different query
        query_sys = (
            "You generate precise DuckDuckGo search queries (4-7 words). "
            "Return ONLY the query string — no quotes, no boolean operators."
        )
        if retry_count > 0:
            query_sys += (
                " This is a RETRY. Generate a MORE SPECIFIC query than before. "
                "Focus on data, statistics, or named strategies."
            )

        query_response = llm.invoke([
            SystemMessage(content=query_sys),
            HumanMessage(content=f"Create search query for: {task['task']}"),
        ])
        search_query = query_response.content.strip().strip('"')
        print(f"   🔎 Query: '{search_query}'")

        # Step 2: Run search
        try:
            search_results = search.run(search_query)
        except Exception as search_err:
            search_results = f"Search failed: {search_err}"
            print(f"   ⚠️  Search error: {search_err}")

        # Step 3: Summarize
        summary_response = llm.invoke([
            SystemMessage(content=(
                "You are a senior marketing research analyst. "
                "Summarize search results to directly answer the task. "
                "Include SPECIFIC data: percentages, statistics, named tools/strategies. "
                "Format:\n"
                "TASK: [restate task]\n"
                "FINDINGS:\n• [specific finding with data]\n• [another finding]"
                f"{retry_note}"
            )),
            HumanMessage(content=(
                f"Task: {task['task']}\n\n"
                f"Search Results:\n{search_results}"
            )),
        ])

        all_findings.append(summary_response.content.strip())
        print(f"   ✅ Task {task['id']} researched.")

        if len(state["external_tasks"]) > 1:
            time.sleep(1.5)

    combined = "\n\n" + "─" * 50 + "\n\n".join(all_findings)
    return {
        "research_output": f"=== 🌐 Market Research Findings ===\n{combined}",
        "research_retry_count": retry_count + 1,   # ★ increment
    }