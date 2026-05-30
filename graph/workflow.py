# =============================================================
# graph/workflow.py — The LangGraph State Machine (Phase 2)
# =============================================================
# WHAT CHANGED FROM PHASE 1:
#   ★ memory_agent node added at the very START
#   ★ rag_agent node added between research and compiler
#   ★ save_session node added at the END (writes to ChromaDB)
#   ★ route_after_coding now routes to "rag_agent" instead of "compiler"
#     (since rag_agent always runs before compiler now)
#
# NEW COMPLETE FLOW:
#   START
#     ↓
#   [memory_agent]     ← ★ NEW: retrieves past context into state
#     ↓
#   [planner]          ← reads memory_context, creates plan
#     ↓ (conditional)
#   ┌──────────────────────────────────────────────┐
#   │  has internal tasks + CSV → [coding_agent]   │
#   │  else                     → [research_agent] │
#   └──────────────────────────────────────────────┘
#     ↓ (conditional after coding)
#   ┌────────────────────────────────────────────────┐
#   │  has external tasks → [research_agent]         │
#   │  else               → [rag_agent]  ← ★ changed │
#   └────────────────────────────────────────────────┘
#     ↓ (research always goes to rag_agent)
#   [rag_agent]        ← ★ NEW: retrieves historical comparisons
#     ↓
#   [compiler]         ← reads everything: data, research, memory, rag
#     ↓
#   [save_session]     ← ★ NEW: persists session to ChromaDB
#     ↓
#   END
# =============================================================

from langgraph.graph import END, START, StateGraph

from agents.coding_agent import coding_agent
from agents.compiler import compiler_agent
from agents.memory_agent import memory_agent
from agents.planner import planner_agent
from agents.rag_agent import rag_agent
from agents.research_agent import research_agent
from memory import store
from state import MarketingState


# ──────────────────────────────────────────────────────────────
# ROUTING FUNCTIONS
# ──────────────────────────────────────────────────────────────

def route_after_planner(state: MarketingState) -> str:
    """After Planner: go to Coding Agent or skip to Research?"""
    has_internal = bool(state.get("internal_tasks"))
    has_data = bool(state.get("data_path"))

    if has_internal and has_data:
        print("   🔀 [ROUTER] Planner → Coding Agent")
        return "coding_agent"
    else:
        print("   🔀 [ROUTER] Planner → Research Agent")
        return "research_agent"


def route_after_coding(state: MarketingState) -> str:
    """
    After Coding Agent: go to Research Agent or skip to RAG Agent?
    (Phase 2: skip target changed from "compiler" to "rag_agent")
    """
    has_external = bool(state.get("external_tasks"))

    if has_external:
        print("   🔀 [ROUTER] Coding Agent → Research Agent")
        return "research_agent"
    else:
        print("   🔀 [ROUTER] Coding Agent → RAG Agent (no web research needed)")
        return "rag_agent"


# ──────────────────────────────────────────────────────────────
# SAVE SESSION NODE
# Defined here (not in agents/) because it's more of a graph
# lifecycle step than a true "agent" — it has no LLM call.
# ──────────────────────────────────────────────────────────────

def save_session_node(state: MarketingState) -> dict:
    """
    Save Session Node.

    Runs after compiler. Persists the full session to ChromaDB
    so future runs can retrieve it via memory_agent and rag_agent.

    Reads:  state["query"], state["coding_output"],
            state["research_output"], state["final_report"]
    Writes: nothing to state (returns {} — state unchanged)
    """
    print("\n💾 [SAVE SESSION] Persisting session to memory store...")

    # Only save if we have a completed report
    final_report = state.get("final_report", "")
    if not final_report or len(final_report) < 50:
        print("   ⚠️  Report too short or empty — skipping save.")
        return {}

    try:
        session_id = store.save_session(
            query=state["query"],
            coding_output=state.get("coding_output", ""),
            research_output=state.get("research_output", ""),
            final_report=final_report,
        )
        print(f"   ✅ Saved as: {session_id}")
        total = store.session_count()
        print(f"   📚 Total sessions in memory: {total}")
    except Exception as e:
        # Never let a save failure crash the app — just log it
        print(f"   ❌ Failed to save session: {e}")

    return {}  # No state updates — session is already complete


# ──────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ──────────────────────────────────────────────────────────────

def build_graph():
    """
    Builds and compiles the Phase 2 LangGraph StateGraph.

    Returns:
        CompiledStateGraph: Ready to use with .stream() or .invoke()
    """
    graph = StateGraph(MarketingState)

    # ── Register all nodes ─────────────────────────────────────
    graph.add_node("memory_agent",   memory_agent)    # ★ Phase 2
    graph.add_node("planner",        planner_agent)
    graph.add_node("coding_agent",   coding_agent)
    graph.add_node("research_agent", research_agent)
    graph.add_node("rag_agent",      rag_agent)       # ★ Phase 2
    graph.add_node("compiler",       compiler_agent)
    graph.add_node("save_session",   save_session_node)  # ★ Phase 2

    # ── Edges ──────────────────────────────────────────────────

    # ★ Pipeline now starts with memory_agent (not planner)
    graph.add_edge(START, "memory_agent")

    # memory_agent always goes to planner
    graph.add_edge("memory_agent", "planner")

    # Planner: conditional routing (same as Phase 1)
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "coding_agent":   "coding_agent",
            "research_agent": "research_agent",
        },
    )

    # Coding Agent: conditional routing
    # ★ Changed: fallback now goes to "rag_agent" instead of "compiler"
    graph.add_conditional_edges(
        "coding_agent",
        route_after_coding,
        {
            "research_agent": "research_agent",
            "rag_agent":      "rag_agent",      # ★ changed from "compiler"
        },
    )

    # Research always goes to rag_agent (★ changed from "compiler")
    graph.add_edge("research_agent", "rag_agent")

    # ★ rag_agent always goes to compiler
    graph.add_edge("rag_agent", "compiler")

    # ★ compiler goes to save_session (not END)
    graph.add_edge("compiler", "save_session")

    # ★ save_session is the new terminal node
    graph.add_edge("save_session", END)

    compiled = graph.compile()
    print("✅ [GRAPH] Phase 2 LangGraph pipeline compiled successfully.")
    return compiled