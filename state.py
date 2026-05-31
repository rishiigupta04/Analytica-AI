# =============================================================
# state.py — The Shared State (Phase 3 — updated)
# =============================================================
# WHAT CHANGED FROM PHASE 2:
#   Phase 3 adds 8 new fields to support:
#     ★ HITL (Human-in-the-Loop) plan approval gate
#     ★ Critic Agent validation + retry logic
#
# FULL FIELD MAP:
#   ┌─ INPUT ─────────────────────────────────────────────────┐
#   │  query, data_path                                        │
#   ├─ PHASE 2: MEMORY ───────────────────────────────────────┤
#   │  memory_context, rag_context                             │
#   ├─ PLANNER OUTPUT ────────────────────────────────────────┤
#   │  plan, internal_tasks, external_tasks                    │
#   ├─ ★ PHASE 3: HITL ───────────────────────────────────────┤
#   │  plan_approved        → set True by HITL node            │
#   ├─ ★ PHASE 3: CRITIC ─────────────────────────────────────┤
#   │  critic_flags         → accumulated issues list          │
#   │  coding_retry_count   → how many times coding ran        │
#   │  research_retry_count → how many times research ran      │
#   │  coding_confidence    → critic score for coding (0-1)    │
#   │  research_confidence  → critic score for research (0-1)  │
#   │  critic_coding_feedback   → hint for coding on retry     │
#   │  critic_research_feedback → hint for research on retry   │
#   ├─ AGENT OUTPUTS ─────────────────────────────────────────┤
#   │  coding_output, research_output                          │
#   ├─ FINAL OUTPUT ──────────────────────────────────────────┤
#   │  final_report                                            │
#   └─ ERROR ─────────────────────────────────────────────────┘
#      error
# =============================================================

from typing import List, Optional, TypedDict


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
    """Complete state flowing through the LangGraph pipeline."""

    # ── INPUT ─────────────────────────────────────────────────
    query: str
    data_path: Optional[str]

    # ── PHASE 2: MEMORY & RAG ─────────────────────────────────
    memory_context: str     # From memory_agent — past session summaries
    rag_context: str        # From rag_agent — historical comparison data

    # ── PLANNER OUTPUT ────────────────────────────────────────
    plan: List[SubTask]
    internal_tasks: List[SubTask]
    external_tasks: List[SubTask]

    # ── ★ PHASE 3: HITL ───────────────────────────────────────
    plan_approved: bool     # True after user approves in HITL gate

    # ── ★ PHASE 3: CRITIC ─────────────────────────────────────
    critic_flags: List[str]        # Accumulated quality issues (shown in report)
    coding_retry_count: int        # Times coding agent has run (0 = first run)
    research_retry_count: int      # Times research agent has run (0 = first run)
    coding_confidence: float       # Critic's quality score for coding (0.0–1.0)
    research_confidence: float     # Critic's quality score for research (0.0–1.0)
    critic_coding_feedback: str    # Specific feedback for coding retry prompt
    critic_research_feedback: str  # Specific feedback for research retry prompt

    # ── AGENT OUTPUTS ─────────────────────────────────────────
    coding_output: str
    research_output: str

    # ── FINAL OUTPUT ──────────────────────────────────────────
    final_report: str

    # ── ERROR TRACKING ────────────────────────────────────────
    error: Optional[str]