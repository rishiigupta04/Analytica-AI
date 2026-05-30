# =============================================================
# agents/research_agent.py — The Research Agent
# =============================================================
# ROLE: Web researcher for external marketing knowledge.
#
# WHAT IT DOES:
#   For each external_task from the Planner:
#     1. Asks the LLM to create an optimized search query
#     2. Runs a Tavily search (requires API key)
#     3. Asks the LLM to summarize the results relevantly
#   Returns a combined research summary string.
#
# WHY TAVILY:
#   - Higher relevance for marketing research
#   - Structured results with titles and URLs
#   - Requires TAVILY_API_KEY
# =============================================================

import os
import time
from langchain_community.tools import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage

from state import MarketingState
from utils.llm import get_llm


def research_agent(state: MarketingState) -> dict:
    """
    Research Agent Node.

    Reads:  state["external_tasks"]
    Writes: state["research_output"]
    """

    # ── Guard: skip if no external research needed ─────────────
    if not state.get("external_tasks"):
        print("   ⏭️  [RESEARCH AGENT] No external tasks — skipping.")
        return {"research_output": "No external research required for this query."}

    if not os.getenv("TAVILY_API_KEY"):
        print("   ⏭️  [RESEARCH AGENT] TAVILY_API_KEY not set — skipping.")
        return {"research_output": "External research skipped: TAVILY_API_KEY is not set."}

    print(f"\n🔍 [RESEARCH AGENT] Researching {len(state['external_tasks'])} task(s)...")

    llm = get_llm(temperature=0)
    search = TavilySearchResults(max_results=6)
    all_findings = []

    def format_search_results(results) -> str:
        if isinstance(results, str):
            return results
        if isinstance(results, dict):
            return "\n".join(f"{key}: {value}" for key, value in results.items())
        if isinstance(results, list):
            lines = []
            for item in results:
                if isinstance(item, dict):
                    title = item.get("title") or item.get("name") or "Result"
                    url = item.get("url") or item.get("link") or ""
                    content = item.get("content") or item.get("snippet") or ""
                    line = f"- {title}\n  {url}\n  {content}".strip()
                    lines.append(line)
                else:
                    lines.append(str(item))
            return "\n".join(lines) if lines else str(results)
        return str(results)

    for task in state["external_tasks"]:
        print(f"   🌐 Researching: {task['task'][:60]}...")

        # ── Step 1: Generate an optimized search query ─────────
        # The task description might be long — we need a tight search query
        query_response = llm.invoke([
            SystemMessage(
                content=(
                    "You generate short, precise search queries for Tavily. "
                    "Return ONLY the query string — 4 to 7 words. "
                    "Focus on the marketing concept being asked about, include a year if recency matters. "
                    "Prefer terms that surface benchmarks, statistics, or best practices. "
                    "Do not add quotes, boolean operators, or site: filters."
                )
            ),
            HumanMessage(content=f"Create a search query for this task:\n{task['task']}"),
        ])
        search_query = query_response.content.strip().strip('"')
        print(f"   🔎 Search query: '{search_query}'")

        # ── Step 2: Run Tavily search ──────────────────────────
        try:
            raw_results = search.run(search_query)
            search_results = format_search_results(raw_results)
        except Exception as search_err:
            search_results = f"Search failed: {search_err}"
            print(f"   ⚠️  Search error: {search_err}")

        # ── Step 3: Summarize and filter results ───────────────
        summary_response = llm.invoke([
            SystemMessage(
                content=(
                    "You are a senior marketing research analyst. "
                    "Summarize the search results below to directly answer the given task. "
                    "Be concise but specific — include actual statistics, years, or source names when present. "
                    "Flag uncertainty explicitly if the evidence is thin or conflicting. "
                    "Format your response as:\n"
                    "TASK: [restate the task in one line]\n"
                    "FINDINGS: [2-4 bullet points of key insights]\n"
                    "If the results don't answer the task well, say so honestly."
                )
            ),
            HumanMessage(
                content=(
                    f"Task: {task['task']}\n\n"
                    f"Search Query Used: {search_query}\n\n"
                    f"Search Results:\n{search_results}"
                )
            ),
        ])

        all_findings.append(summary_response.content.strip())
        print(f"   ✅ Task {task['id']} research complete.")

        # Small delay to avoid rate limiting when multiple tasks run
        if len(state["external_tasks"]) > 1:
            time.sleep(1.5)

    combined = "\n\n" + "─" * 50 + "\n\n".join(all_findings)
    return {
        "research_output": f"=== 🌐 Market Research Findings ===\n{combined}"
    }
