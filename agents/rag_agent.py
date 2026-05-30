# =============================================================
# agents/rag_agent.py — The RAG (Retrieval-Augmented Generation) Agent
# =============================================================
# ROLE: Runs AFTER coding + research agents, BEFORE the compiler.
#       Adds historical depth to the final report.
#
# HOW RAG WORKS:
#   Retrieval-Augmented Generation = search a knowledge base,
#   then feed the retrieved text to an LLM as additional context.
#   In our case: ChromaDB is the knowledge base (past sessions),
#   and the LLM uses retrieved data to enrich the report.
#
# DIFFERENCE FROM MEMORY AGENT:
#   memory_agent → runs at START → helps the PLANNER make a
#                  better plan by knowing what was done before
#
#   rag_agent    → runs at END   → helps the COMPILER write a
#                  richer report by comparing CURRENT findings
#                  against PAST findings from similar sessions
#
# WHAT IT ENABLES:
#   - "Last month, this channel had $43K revenue. Now it's $38K — a 12% drop."
#   - "You previously researched attribution modeling on Jan 15."
#   - "Industry benchmark found 3 sessions ago: 8-12% engagement rate."
#
# WHEN IT RETURNS GENERIC CONTEXT:
#   - No past sessions exist
#   - No sufficiently similar sessions found
#   → Returns a note for the compiler to skip historical comparison
# =============================================================

from langchain_core.messages import HumanMessage, SystemMessage

from memory import store
from state import MarketingState
from utils.llm import get_llm


def rag_agent(state: MarketingState) -> dict:
    """
    RAG Agent Node.

    Reads:  state["query"], state["coding_output"], state["research_output"]
    Writes: state["rag_context"]
    """

    print(f"\n📁 [RAG AGENT] Retrieving historical findings for compiler...")

    # ── Build a rich search query ──────────────────────────────
    # We search using query + current findings for better relevance.
    # This finds sessions that are topically similar AND have
    # similar data characteristics.
    current_findings_snippet = ""
    if state.get("coding_output") and "No internal" not in state["coding_output"]:
        current_findings_snippet = state["coding_output"][:400]
    if state.get("research_output") and "No external" not in state["research_output"]:
        current_findings_snippet += "\n" + state["research_output"][:300]

    search_text = f"{state['query']}\n{current_findings_snippet}"

    # ── Search ChromaDB ────────────────────────────────────────
    similar_sessions = store.search_sessions(search_text, top_k=2)

    if not similar_sessions:
        print("   ℹ️  No historical sessions to retrieve — first run.")
        return {"rag_context": "No historical data available for comparison."}

    # For RAG, we use a stricter relevance threshold than memory agent
    # We only want highly similar sessions to avoid confusing the compiler
    relevant = [s for s in similar_sessions if s["distance"] < 0.35]

    if not relevant:
        print(f"   ℹ️  Closest historical session has distance "
              f"{similar_sessions[0]['distance']:.2f} — not similar enough for RAG.")
        return {"rag_context": "No closely matching historical data found for this query."}

    print(f"   ✅ {len(relevant)} historical session(s) retrieved for RAG.")

    # ── Format retrieved sessions ──────────────────────────────
    retrieved_text = ""
    for i, session in enumerate(relevant, 1):
        meta = session["metadata"]
        retrieved_text += (
            f"\n=== Historical Session {i} ===\n"
            f"Date: {meta.get('timestamp', 'unknown')[:10]}\n"
            f"Original Query: {meta.get('query', 'N/A')}\n"
            f"Findings:\n{session['document'][:700]}\n"
        )

    # ── Ask LLM to create a comparison context ─────────────────
    # The LLM's job here is to extract historical data points
    # that can be compared against the current session's findings.
    # The compiler will receive this as "rag_context".
    llm = get_llm(temperature=0)

    has_current_data = bool(current_findings_snippet.strip())

    rag_prompt = (
        f"Current Query: {state['query']}\n\n"
        f"{'Current Findings (this session):' + current_findings_snippet if has_current_data else 'No current internal data was analyzed.'}\n\n"
        f"Historical Data Retrieved:\n{retrieved_text}\n\n"
        "Extract historical data points that are DIRECTLY comparable to the current session. "
        "Focus on: specific metrics, channel names, percentages, revenue figures, and conclusions. "
        "Format as: '📅 [Date]: [specific finding that can be compared to current data]'"
    )

    rag_response = llm.invoke([
        SystemMessage(content=(
            "You are a data analyst extracting historical comparison points. "
            "Your output will be used by a report compiler to identify trends over time. "
            "Be specific and factual — only include data that actually appears in the historical sessions. "
            "If the historical data isn't comparable to the current query, say so clearly. "
            "Maximum 5 bullet points."
        )),
        HumanMessage(content=rag_prompt),
    ])

    rag_context = (
        f"📊 Historical Comparison Data (from {len(relevant)} past session(s)):\n"
        f"{rag_response.content.strip()}"
    )

    print(f"   ✅ RAG context built ({len(rag_context)} chars).")
    return {"rag_context": rag_context}