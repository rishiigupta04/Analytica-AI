# =============================================================
# agents/memory_agent.py — The Context Memory Agent
# =============================================================
# ROLE: The first agent in Phase 2's pipeline. Runs BEFORE
#       the Planner to give it historical context.
#
# WHAT IT DOES:
#   1. Takes the marketer's current query
#   2. Searches ChromaDB for semantically similar past sessions
#   3. Asks the LLM to distill the past findings into a
#      brief, relevant context summary
#   4. Writes that summary into state["memory_context"]
#   5. The Planner reads memory_context to:
#        - Avoid re-doing research already completed
#        - Reference past findings in its plan
#        - Build on earlier analysis
#
# ANALOGY:
#   This is like a colleague who, before your meeting starts,
#   says: "Oh, we looked at this exact thing last month — here's
#   what we found." The Planner can then say "great, skip that
#   research step, we already have the answer."
#
# WHEN IT RETURNS EMPTY:
#   - First ever run (no sessions stored yet)
#   - No semantically similar past sessions found
#   - In both cases: empty string → Planner gets no extra context
#     and works normally as in Phase 1
# =============================================================

from langchain_core.messages import HumanMessage, SystemMessage

from memory import store
from state import MarketingState
from utils.llm import get_llm


def memory_agent(state: MarketingState) -> dict:
    """
    Context Memory Agent Node.

    Reads:  state["query"]
    Writes: state["memory_context"]
    """

    print(f"\n🧩 [MEMORY AGENT] Searching for relevant past sessions...")

    # ── Step 1: Search the vector store ───────────────────────
    similar_sessions = store.search_sessions(state["query"], top_k=3)

    if not similar_sessions:
        print("   ℹ️  No past sessions found — first run or new topic.")
        return {"memory_context": ""}

    # Filter by relevance — distance < 0.5 means meaningfully similar
    # (cosine distance: 0 = identical, 1 = completely unrelated)
    relevant = [s for s in similar_sessions if s["distance"] < 0.5]

    if not relevant:
        print(f"   ℹ️  {len(similar_sessions)} sessions found but none similar enough "
              f"(closest distance: {similar_sessions[0]['distance']:.2f})")
        return {"memory_context": ""}

    print(f"   ✅ {len(relevant)} relevant past session(s) found.")

    # ── Step 2: Format sessions for LLM ───────────────────────
    sessions_text = ""
    for i, session in enumerate(relevant, 1):
        meta = session["metadata"]
        sessions_text += (
            f"\n--- Past Session {i} ---\n"
            f"When: {meta.get('timestamp', 'unknown')[:10]}\n"
            f"Query: {meta.get('query', 'N/A')}\n"
            f"Content:\n{session['document'][:600]}\n"
        )

    # ── Step 3: Ask LLM to summarize what's actually relevant ─
    # We don't just dump all past sessions — we ask the LLM to
    # pick out what's useful for THIS current question
    llm = get_llm(temperature=0)

    summary_response = llm.invoke([
        SystemMessage(content=(
            "You are a marketing analyst reviewing past research sessions. "
            "Your job is to extract ONLY what's directly relevant to the current query. "
            "Be concise — 3 to 6 bullet points maximum. "
            "Focus on: past data findings, metrics found, conclusions drawn, research done. "
            "If the past sessions aren't relevant to the current query, respond with: NONE\n"
            "Format:\n"
            "• [Finding or context point from past session]\n"
            "• [Another relevant point]\n"
            "Note: These are from past analyses — not necessarily still current."
        )),
        HumanMessage(content=(
            f"Current Query: {state['query']}\n\n"
            f"Past Sessions:\n{sessions_text}\n\n"
            "Extract only what's relevant to the current query:"
        )),
    ])

    raw_summary = summary_response.content.strip()

    if raw_summary.upper() == "NONE" or not raw_summary:
        print("   ℹ️  Past sessions found but not relevant to current query.")
        return {"memory_context": ""}

    memory_context = (
        f"📚 Context from {len(relevant)} similar past session(s):\n{raw_summary}"
    )

    print(f"   ✅ Memory context ready ({len(memory_context)} chars).")
    return {"memory_context": memory_context}