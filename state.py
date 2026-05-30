# =============================================================
# state.py — The Shared State (Phase 2 — updated)
# =============================================================
# WHAT CHANGED FROM PHASE 1:
#   Two new fields added:
#     memory_context → written by memory_agent (before planner)
#     rag_context    → written by rag_agent (before compiler)
#
# These two fields carry historical intelligence from ChromaDB
# into the current pipeline run. All other fields unchanged.
# =============================================================

from typing import TypedDict, List, Optional


class SubTask(TypedDict):
    """
    One specific actionable task from the Planner.

    source:
        "INTERNAL"  → analyze uploaded CSV data
        "EXTERNAL"  → search the web
        "BOTH"      → do both
    """
    id: str
    task: str
    source: str


class MarketingState(TypedDict):
    """
    Complete state object flowing through the LangGraph pipeline.

    Phase 2 additions marked with ★
    """

    # ── INPUT ─────────────────────────────────────────────────
    query: str                  # The marketer's original question
    data_path: Optional[str]    # Path to uploaded CSV (or None)

    # ── ★ PHASE 2: MEMORY & RAG ───────────────────────────────
    memory_context: str         # Written by memory_agent BEFORE planner
                                # Contains: relevant past session summaries
                                # Used by: planner (better planning) + compiler

    rag_context: str            # Written by rag_agent BEFORE compiler
                                # Contains: historical comparison data points
                                # Used by: compiler (trend analysis in report)

    # ── PLANNER OUTPUT ────────────────────────────────────────
    plan: List[SubTask]             # Full decomposed plan
    internal_tasks: List[SubTask]   # Tasks needing CSV analysis
    external_tasks: List[SubTask]   # Tasks needing web research

    # ── AGENT OUTPUTS ─────────────────────────────────────────
    coding_output: str      # Coding Agent's data analysis results
    research_output: str    # Research Agent's web findings

    # ── FINAL OUTPUT ──────────────────────────────────────────
    final_report: str       # Compiled report shown to marketer

    # ── ERROR TRACKING ────────────────────────────────────────
    error: Optional[str]    # Any error message (None = no error)