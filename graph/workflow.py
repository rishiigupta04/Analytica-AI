# =============================================================
# graph/workflow.py — The LangGraph State Machine (Phase 3)
# =============================================================
# WHAT CHANGED FROM PHASE 2:
#   ★ HITL (Human-in-the-Loop) node using interrupt()
#   ★ Critic nodes: critique_coding + critique_research
#   ★ Retry routing: agents can loop back up to 2x if critic fails
#   ★ MemorySaver checkpointer enables graph state persistence
#     between the pre-HITL and post-HITL Streamlit runs
#   ★ graph.compile(checkpointer=MemorySaver()) — required for interrupt()
#
# COMPLETE PHASE 3 FLOW:
#
#   START
#     ↓
#   [memory_agent]            ← retrieves past context
#     ↓
#   [planner]                 ← decomposes query + uses memory
#     ↓
#   ★ [hitl_node]             ← interrupt() fires here
#     │                          graph PAUSES, Streamlit shows plan
#     │                          user approves/edits/removes tasks
#     │                          graph RESUMES with Command(resume=...)
#     ↓
#   conditional ──────────────────────────────────────────────
#     │  has internal + CSV  → [coding_agent]
#     └  else                → [research_agent]
#              ↓
#          ★ [critique_coding]
#              │  conf<0.6 AND retries<2  → [coding_agent] (retry)
#              └  else                    → [research_agent] or [rag_agent]
#                       ↓
#                   [research_agent]
#                       ↓
#                ★ [critique_research]
#                       │  conf<0.6 AND retries<2  → [research_agent] (retry)
#                       └  else                    → [rag_agent]
#                                ↓
#                           [rag_agent]
#                                ↓
#                           [compiler]          ← Pydantic structured output
#                                ↓
#                           [save_session]      ← persists to ChromaDB
#                                ↓
#                             END
# =============================================================

import uuid
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

from agents.coding_agent import coding_agent
from agents.compiler import compiler_agent
from agents.critic_agent import critique_coding, critique_research
from agents.memory_agent import memory_agent
from agents.planner import planner_agent
from agents.rag_agent import rag_agent
from agents.research_agent import research_agent
from memory import store
from state import MarketingState


# ──────────────────────────────────────────────────────────────
# ★ HITL NODE
# ──────────────────────────────────────────────────────────────

def hitl_node(state: MarketingState) -> dict:
    """
    Human-in-the-Loop Gate Node.

    Calls interrupt() which PAUSES graph execution and hands
    control back to the calling code (app.py).

    app.py receives the interrupt payload, shows it in the UI,
    collects user input, then resumes the graph with:
        graph.stream(Command(resume={"plan": approved_plan}), config=config)

    The hitl_node then receives that resume value and uses it
    to update the state before the graph continues.

    Reads:  state["plan"], state["query"], state["memory_context"]
    Writes: state["plan"], state["internal_tasks"],
            state["external_tasks"], state["plan_approved"]
    """
    print("\n🧑‍✈️ [HITL] Pausing for human plan review...")

    # interrupt() pauses the graph and sends this payload to the caller.
    # The caller receives it in the __interrupt__ event during streaming.
    user_response = interrupt({
        "plan": state["plan"],
        "query": state["query"],
        "memory_context": state.get("memory_context", ""),
        "has_data": bool(state.get("data_path")),
        "message": "Review the plan below. You may edit tasks or their sources before approving.",
    })

    # user_response = {"plan": [...potentially edited plan...]}
    # This value comes from Command(resume={...}) in app.py
    approved_plan = user_response.get("plan", state["plan"])
    has_data = bool(state.get("data_path"))

    # Recompute routing lists from the user-approved (possibly edited) plan
    internal_tasks = [
        t for t in approved_plan
        if t["source"] in ("INTERNAL", "BOTH") and has_data
    ]
    external_tasks = [
        t for t in approved_plan
        if t["source"] in ("EXTERNAL", "BOTH")
    ]

    print(f"   ✅ Plan approved: {len(approved_plan)} task(s) "
          f"({len(internal_tasks)} internal, {len(external_tasks)} external)")

    return {
        "plan": approved_plan,
        "internal_tasks": internal_tasks,
        "external_tasks": external_tasks,
        "plan_approved": True,
    }


# ──────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# ──────────────────────────────────────────────────────────────

def route_after_hitl(state: MarketingState) -> str:
    """After HITL approval: coding or research?"""
    has_internal = bool(state.get("internal_tasks"))
    has_data = bool(state.get("data_path"))

    if has_internal and has_data:
        print("   🔀 [ROUTER] HITL → Coding Agent")
        return "coding_agent"
    else:
        print("   🔀 [ROUTER] HITL → Research Agent")
        return "research_agent"


def route_after_coding_critique(state: MarketingState) -> str:
    """
    After Coding Critique:
      - Low confidence + retries left  → retry coding_agent
      - Otherwise                       → research_agent or rag_agent
    """
    confidence  = state.get("coding_confidence", 1.0)
    retry_count = state.get("coding_retry_count", 0)
    has_external = bool(state.get("external_tasks"))

    if confidence < 0.6 and retry_count < 2:
        print(f"   🔀 [ROUTER] Critique → Coding Retry "
              f"(confidence={confidence:.0%}, attempt {retry_count+1}/3)")
        return "coding_agent"

    if confidence < 0.6:
        print(f"   🔀 [ROUTER] Critique → Proceeding despite low confidence "
              f"(max retries reached)")
    else:
        print(f"   🔀 [ROUTER] Critique → Passed (confidence={confidence:.0%})")

    if has_external:
        return "research_agent"
    return "rag_agent"


def route_after_research_critique(state: MarketingState) -> str:
    """
    After Research Critique:
      - Low confidence + retries left  → retry research_agent
      - Otherwise                       → rag_agent
    """
    confidence  = state.get("research_confidence", 1.0)
    retry_count = state.get("research_retry_count", 0)

    if confidence < 0.6 and retry_count < 2:
        print(f"   🔀 [ROUTER] Critique → Research Retry "
              f"(confidence={confidence:.0%}, attempt {retry_count+1}/3)")
        return "research_agent"

    if confidence < 0.6:
        print(f"   🔀 [ROUTER] Critique → Proceeding despite low confidence")
    else:
        print(f"   🔀 [ROUTER] Critique → Passed (confidence={confidence:.0%})")

    return "rag_agent"


# ──────────────────────────────────────────────────────────────
# SAVE SESSION NODE
# ──────────────────────────────────────────────────────────────

def save_session_node(state: MarketingState) -> dict:
    """Persists completed session to ChromaDB. Terminal node."""
    print("\n💾 [SAVE SESSION] Persisting session to memory store...")
    final_report = state.get("final_report", "")

    if not final_report or len(final_report) < 50:
        print("   ⚠️  Report too short — skipping save.")
        return {}

    try:
        session_id = store.save_session(
            query=state["query"],
            coding_output=state.get("coding_output", ""),
            research_output=state.get("research_output", ""),
            final_report=final_report,
        )
        total = store.session_count()
        print(f"   ✅ Saved as: {session_id} (total sessions: {total})")
    except Exception as e:
        print(f"   ❌ Save failed: {e}")

    return {}


# ──────────────────────────────────────────────────────────────
# ★ GRAPH BUILDER — with MemorySaver for HITL interrupt support
# ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Builds the Phase 3 LangGraph pipeline.

    KEY DIFFERENCE from Phase 2:
        graph.compile(checkpointer=MemorySaver())

    The MemorySaver stores graph state snapshots indexed by thread_id.
    When interrupt() fires and the graph pauses, the MemorySaver
    holds the full state. When Command(resume=...) is called later
    with the same thread_id, the graph resumes from that snapshot.

    IMPORTANT for Streamlit: Cache the compiled graph in
    st.session_state so the same MemorySaver instance is reused
    across Streamlit re-runs (otherwise the checkpoint is lost).

    Returns:
        CompiledStateGraph: Ready to use with .stream()
    """
    graph = StateGraph(MarketingState)

    # ── Register nodes ─────────────────────────────────────────
    graph.add_node("memory_agent",       memory_agent)
    graph.add_node("planner",            planner_agent)
    graph.add_node("hitl_node",          hitl_node)         # ★ Phase 3
    graph.add_node("coding_agent",       coding_agent)
    graph.add_node("critique_coding",    critique_coding)   # ★ Phase 3
    graph.add_node("research_agent",     research_agent)
    graph.add_node("critique_research",  critique_research) # ★ Phase 3
    graph.add_node("rag_agent",          rag_agent)
    graph.add_node("compiler",           compiler_agent)
    graph.add_node("save_session",       save_session_node)

    # ── Edges ──────────────────────────────────────────────────
    graph.add_edge(START, "memory_agent")
    graph.add_edge("memory_agent", "planner")

    # ★ Planner now always goes to HITL (not directly to routing)
    graph.add_edge("planner", "hitl_node")

    # After HITL: route to coding or research
    graph.add_conditional_edges(
        "hitl_node",
        route_after_hitl,
        {
            "coding_agent":   "coding_agent",
            "research_agent": "research_agent",
        },
    )

    # After coding: always critique
    graph.add_edge("coding_agent", "critique_coding")

    # After critique_coding: retry, continue to research, or skip to rag
    graph.add_conditional_edges(
        "critique_coding",
        route_after_coding_critique,
        {
            "coding_agent":   "coding_agent",    # retry
            "research_agent": "research_agent",  # proceed with research
            "rag_agent":      "rag_agent",        # skip research
        },
    )

    # After research: always critique
    graph.add_edge("research_agent", "critique_research")

    # After critique_research: retry or proceed to rag
    graph.add_conditional_edges(
        "critique_research",
        route_after_research_critique,
        {
            "research_agent": "research_agent",  # retry
            "rag_agent":      "rag_agent",        # proceed
        },
    )

    graph.add_edge("rag_agent",      "compiler")
    graph.add_edge("compiler",       "save_session")
    graph.add_edge("save_session",   END)

    # ★ MemorySaver: required for interrupt() / HITL to work
    compiled = graph.compile(checkpointer=MemorySaver())
    print("✅ [GRAPH] Phase 3 LangGraph pipeline compiled with MemorySaver.")
    return compiled