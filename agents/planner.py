# =============================================================
# agents/planner.py — The Planner Agent (Phase 2 — updated)
# =============================================================
# WHAT CHANGED FROM PHASE 1:
#   ★ Reads state["memory_context"] (set by memory_agent)
#   ★ Includes past context in the LLM prompt so the Planner
#     can build on what was already researched, avoid duplicate
#     work, and reference past data in its plan
#
# Example impact:
#   Phase 1 Planner might plan: "Research what email marketing is"
#   Phase 2 Planner reads memory_context and sees: "We researched
#   email marketing 2 days ago — findings were X" → skips that
#   research task or refines it to build on the existing answer
# =============================================================

import json
from langchain_core.messages import HumanMessage, SystemMessage

from state import MarketingState, SubTask
from utils.llm import get_llm


PLANNER_SYSTEM_PROMPT = """You are a planning agent for a marketing Business Intelligence assistant.

A marketer will ask you a question. Your job is to:
1. Break it into specific, answerable sub-tasks (max 4)
2. Tag each sub-task with the correct data source

DATA SOURCE RULES (follow these strictly):
- INTERNAL → The answer requires analyzing numbers/metrics from the uploaded CSV file
- EXTERNAL → The answer requires web research about marketing concepts, trends, or best practices
- BOTH     → The answer needs BOTH internal data AND external context

IMPORTANT:
- If no internal data is available, NEVER use INTERNAL or BOTH — use EXTERNAL only
- If memory_context is provided, use it to avoid re-researching already-answered questions
- Be specific: "Find the channel with lowest revenue using Revenue column" not "analyze channels"

Return ONLY valid JSON — no explanation, no markdown, just raw JSON:
{
  "sub_tasks": [
    {"id": "1", "task": "Specific actionable task description", "source": "INTERNAL"},
    {"id": "2", "task": "Another specific task", "source": "EXTERNAL"}
  ]
}"""


def planner_agent(state: MarketingState) -> dict:
    """
    Planner Agent Node.

    Reads:  state["query"], state["data_path"], state["memory_context"] ★
    Writes: state["plan"], state["internal_tasks"], state["external_tasks"]
    """
    llm = get_llm(temperature=0)

    has_data = bool(state.get("data_path"))
    data_context = (
        "Internal CSV data IS available — you may use INTERNAL or BOTH sources."
        if has_data
        else "No internal CSV data uploaded — use EXTERNAL sources only."
    )

    # ★ Phase 2: include memory context in the planning prompt
    memory_context = state.get("memory_context", "")
    memory_section = ""
    if memory_context:
        memory_section = (
            f"\n\nPAST SESSION CONTEXT (use this to avoid duplicate work):\n"
            f"{memory_context}\n"
            f"If a task above was already answered in a past session, "
            f"you may still include it but note it can be answered quickly from memory."
        )

    user_message = (
        f"Marketer's Query: {state['query']}\n\n"
        f"Data availability: {data_context}"
        f"{memory_section}\n\n"
        f"Create a clear plan with sub-tasks to fully answer the marketer's query."
    )

    print(f"\n🧠 [PLANNER] Analyzing query: {state['query'][:80]}...")
    if memory_context:
        print(f"   📚 Using memory context ({len(memory_context)} chars)")

    response = llm.invoke([
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    # ── Parse JSON response ────────────────────────────────────
    try:
        raw = response.content.strip()
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()

        plan_data = json.loads(raw)
        sub_tasks: list[SubTask] = plan_data["sub_tasks"]

        if not has_data:
            for task in sub_tasks:
                if task["source"] in ("INTERNAL", "BOTH"):
                    task["source"] = "EXTERNAL"

        internal_tasks = [t for t in sub_tasks if t["source"] in ("INTERNAL", "BOTH")]
        external_tasks = [t for t in sub_tasks if t["source"] in ("EXTERNAL", "BOTH")]

        print(f"   ✅ Plan: {len(sub_tasks)} sub-tasks "
              f"({len(internal_tasks)} internal, {len(external_tasks)} external)")

        return {
            "plan": sub_tasks,
            "internal_tasks": internal_tasks,
            "external_tasks": external_tasks,
            "coding_output": "",
            "research_output": "",
            "error": None,
        }

    except (json.JSONDecodeError, KeyError) as e:
        print(f"   ❌ Planner parse error: {e} — using fallback")
        fallback_task: SubTask = {
            "id": "1",
            "task": state["query"],
            "source": "EXTERNAL" if not has_data else "BOTH",
        }
        return {
            "plan": [fallback_task],
            "internal_tasks": [fallback_task] if has_data else [],
            "external_tasks": [fallback_task],
            "coding_output": "",
            "research_output": "",
            "error": f"Planner parse error (fallback): {str(e)}",
        }