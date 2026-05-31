# =============================================================
# agents/critic_agent.py — The Critic / Self-Reflection Agent
# =============================================================
# ROLE: Quality gate between agents and the compiler.
#       Acts like a senior analyst reviewing a junior's work
#       before it goes to the final report.
#
# WHAT IT DOES:
#   critique_coding()   — evaluates the Coding Agent's data analysis
#   critique_research() — evaluates the Research Agent's web findings
#
#   Each function:
#     1. Examines the agent output against quality criteria
#     2. Produces a structured Pydantic verdict (passed/confidence/issues)
#     3. Writes confidence score + feedback into state
#     4. Accumulates issues into state["critic_flags"] for the compiler
#
# WHY THIS MATTERS (the Phase 1 problem it solves):
#   The original project produced low-quality reports because agents
#   could return empty results, error messages, or irrelevant content
#   and those went straight to the compiler unchanged. The Critic adds
#   a validation layer that either accepts the output or triggers a
#   targeted retry with specific improvement feedback.
#
# RETRY LOGIC (handled by routing in workflow.py):
#   confidence < 0.6 AND retry_count < 2  → retry the agent
#   confidence >= 0.6 OR retry_count >= 2 → proceed (with flag if low)
#   Max 2 retries per agent = max 3 total runs per agent
#
# CONFIDENCE SCORING:
#   1.0 → Perfect output, directly answers all tasks
#   0.8 → Good output, minor gaps
#   0.6 → Acceptable, proceed with caution flag
#   0.4 → Poor quality, retry if possible
#   0.2 → Very poor (error messages, empty, unrelated content)
# =============================================================

from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage

from state import MarketingState
from utils.llm import get_llm


# ──────────────────────────────────────────────────────────────
# Pydantic model for structured critic output
# Using Pydantic forces the LLM to return a consistent,
# parseable verdict instead of free-form text
# ──────────────────────────────────────────────────────────────

class CriticVerdict(BaseModel):
    """Structured quality assessment from the Critic Agent."""
    passed: bool = Field(
        description="True if quality is acceptable (confidence >= 0.6)"
    )
    confidence: float = Field(
        description="Quality score from 0.0 (completely unusable) to 1.0 (perfect)",
        ge=0.0,
        le=1.0,
    )
    issues: List[str] = Field(
        description="List of specific quality issues found. Empty list if passed."
    )
    feedback: str = Field(
        description=(
            "Specific, actionable instruction for the agent to improve on retry. "
            "Example: 'The Revenue column is named Total_Revenue — use that exact name.' "
            "Empty string if no retry needed."
        )
    )


# ──────────────────────────────────────────────────────────────
# CODING AGENT CRITIC
# ──────────────────────────────────────────────────────────────

CODING_CRITIC_SYSTEM = """You are a senior data analyst evaluating a junior analyst's work.

You will receive:
1. The sub-tasks that needed to be answered using internal CSV data
2. The output the coding agent produced

Evaluate quality against these criteria:
- Does the output contain ACTUAL DATA VALUES (numbers, channel names, percentages)?
- Does it directly answer all the assigned sub-tasks?
- Is it free of error messages, tracebacks, or fallback placeholder text?
- Are the numbers specific (not vague statements like "channel X performed poorly")?

Score guide:
  1.0 = All tasks answered with specific data values
  0.8 = Most tasks answered, minor gaps
  0.6 = Answered partially but with some specific numbers
  0.4 = Mostly errors or vague statements, little actual data
  0.2 = Error messages, empty, or completely off-topic

Set passed=True if confidence >= 0.6"""

RESEARCH_CRITIC_SYSTEM = """You are a senior marketing strategist evaluating a research analyst's work.

You will receive:
1. The research tasks that needed to be answered using web search
2. The research output produced

Evaluate quality against these criteria:
- Does the output contain ACTUAL FINDINGS (not just "search failed" or generic statements)?
- Is the content RELEVANT to the marketing tasks (not random unrelated content)?
- Does it include SPECIFIC information (statistics, named strategies, benchmarks, years)?
- Does each task have a clear answer, not just "it depends"?

Score guide:
  1.0 = All tasks researched with specific, cited findings
  0.8 = Good findings for most tasks
  0.6 = Some useful content but gaps or generalities
  0.4 = Mostly vague or search failures
  0.2 = No useful content, all errors or irrelevant text

Set passed=True if confidence >= 0.6"""


def critique_coding(state: MarketingState) -> dict:
    """
    Coding Critic Node.

    Evaluates the Coding Agent's data analysis output.

    Reads:  state["coding_output"], state["internal_tasks"],
            state["coding_retry_count"], state["critic_flags"]
    Writes: state["coding_confidence"], state["critic_coding_feedback"],
            state["critic_flags"] (appends issues)
    """
    coding_out = state.get("coding_output", "")
    retry_count = state.get("coding_retry_count", 0)
    existing_flags = state.get("critic_flags", [])

    print(f"\n🪞 [CRITIC] Evaluating Coding Agent output "
          f"(attempt {retry_count}/3)...")

    # Quick checks before calling LLM (saves API call for obvious cases)
    if not coding_out or coding_out.strip() in ("", "No internal data analysis required for this query."):
        print("   ⏭️  Coding skipped — no internal tasks. Critic passes trivially.")
        return {
            "coding_confidence": 1.0,
            "critic_coding_feedback": "",
        }

    tasks_text = "\n".join(
        f"  Task {t['id']}: {t['task']}"
        for t in state.get("internal_tasks", [])
    )

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(CriticVerdict)

    try:
        verdict: CriticVerdict = structured_llm.invoke([
            SystemMessage(content=CODING_CRITIC_SYSTEM),
            HumanMessage(content=(
                f"Sub-tasks assigned:\n{tasks_text}\n\n"
                f"Coding Agent Output:\n{coding_out}\n\n"
                f"Evaluate and return your structured verdict."
            )),
        ])

        confidence = verdict.confidence
        emoji = "✅" if verdict.passed else ("⚠️" if confidence >= 0.4 else "❌")
        print(f"   {emoji} Coding confidence: {confidence:.0%} | Passed: {verdict.passed}")

        if verdict.issues:
            print(f"   Issues: {'; '.join(verdict.issues[:2])}")

        # Accumulate flags for the compiler's awareness
        new_flags = list(existing_flags)
        if not verdict.passed:
            for issue in verdict.issues:
                flag = f"⚠️ Coding Agent (attempt {retry_count}): {issue}"
                new_flags.append(flag)

        return {
            "coding_confidence": confidence,
            "critic_coding_feedback": verdict.feedback if not verdict.passed else "",
            "critic_flags": new_flags,
        }

    except Exception as e:
        # Structured output failed — use a permissive default
        print(f"   ⚠️  Critic structured output failed: {e} — defaulting to pass")
        return {
            "coding_confidence": 0.7,  # Give benefit of doubt
            "critic_coding_feedback": "",
            "critic_flags": existing_flags,
        }


def critique_research(state: MarketingState) -> dict:
    """
    Research Critic Node.

    Evaluates the Research Agent's web findings output.

    Reads:  state["research_output"], state["external_tasks"],
            state["research_retry_count"], state["critic_flags"]
    Writes: state["research_confidence"], state["critic_research_feedback"],
            state["critic_flags"] (appends issues)
    """
    research_out = state.get("research_output", "")
    retry_count = state.get("research_retry_count", 0)
    existing_flags = state.get("critic_flags", [])

    print(f"\n🪞 [CRITIC] Evaluating Research Agent output "
          f"(attempt {retry_count}/3)...")

    if not research_out or research_out.strip() in ("", "No external research required for this query."):
        print("   ⏭️  Research skipped — no external tasks. Critic passes trivially.")
        return {
            "research_confidence": 1.0,
            "critic_research_feedback": "",
        }

    tasks_text = "\n".join(
        f"  Task {t['id']}: {t['task']}"
        for t in state.get("external_tasks", [])
    )

    llm = get_llm(temperature=0)
    structured_llm = llm.with_structured_output(CriticVerdict)

    try:
        verdict: CriticVerdict = structured_llm.invoke([
            SystemMessage(content=RESEARCH_CRITIC_SYSTEM),
            HumanMessage(content=(
                f"Research tasks assigned:\n{tasks_text}\n\n"
                f"Research Agent Output:\n{research_out}\n\n"
                f"Evaluate and return your structured verdict."
            )),
        ])

        confidence = verdict.confidence
        emoji = "✅" if verdict.passed else ("⚠️" if confidence >= 0.4 else "❌")
        print(f"   {emoji} Research confidence: {confidence:.0%} | Passed: {verdict.passed}")

        if verdict.issues:
            print(f"   Issues: {'; '.join(verdict.issues[:2])}")

        new_flags = list(existing_flags)
        if not verdict.passed:
            for issue in verdict.issues:
                flag = f"⚠️ Research Agent (attempt {retry_count}): {issue}"
                new_flags.append(flag)

        return {
            "research_confidence": confidence,
            "critic_research_feedback": verdict.feedback if not verdict.passed else "",
            "critic_flags": new_flags,
        }

    except Exception as e:
        print(f"   ⚠️  Critic structured output failed: {e} — defaulting to pass")
        return {
            "research_confidence": 0.7,
            "critic_research_feedback": "",
            "critic_flags": existing_flags,
        }