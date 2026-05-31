# =============================================================
# agents/compiler.py — The Compiler Agent (Phase 3 — updated)
# =============================================================
# WHAT CHANGED FROM PHASE 2:
#   ★ Uses Pydantic structured output via llm.with_structured_output()
#   ★ MarketingReport Pydantic model enforces required sections
#   ★ format_report_to_markdown() converts structured obj → Markdown
#   ★ Confidence score shown in report footer
#   ★ Critic flags shown as warnings at report top
#   ★ Fallback to Phase 2 unstructured approach if Pydantic fails
#
# WHY PYDANTIC STRUCTURED OUTPUT:
#   Phase 2 gave the LLM a Markdown template and hoped it would
#   follow it. Sometimes it didn't — missing sections, wrong headers,
#   inconsistent formatting. Pydantic fixes this by forcing the LLM
#   to fill named typed fields, validated by the schema. We then
#   render those fields into Markdown ourselves — guaranteed format.
# =============================================================

from pydantic import BaseModel, Field
from typing import List
from langchain_core.messages import HumanMessage, SystemMessage

from state import MarketingState
from utils.llm import get_llm


# ──────────────────────────────────────────────────────────────
# ★ Phase 3: Pydantic schema for the final report
# Each field is validated by type — the LLM MUST return all of
# these or the structured output call raises an exception
# (which we catch and fall back from gracefully)
# ──────────────────────────────────────────────────────────────

class MarketingReport(BaseModel):
    """Validated structured output schema for the marketing intelligence report."""

    original_request: str = Field(
        description="One sentence restating the marketer's original question"
    )
    analysis_performed: List[str] = Field(
        description="Bullet points listing each sub-task that was completed"
    )
    internal_findings: str = Field(
        description=(
            "Specific data findings from CSV analysis with exact numbers, "
            "channel names, percentages. Write 'Not applicable.' if no internal data."
        )
    )
    external_insights: str = Field(
        description=(
            "Key trends, benchmarks, and best practices from web research. "
            "Include specific statistics or named strategies. "
            "Write 'Not applicable.' if no research was performed."
        )
    )
    historical_trends: str = Field(
        description=(
            "Comparison with past sessions from RAG historical data. "
            "Show metric changes over time if available. "
            "Write 'No previous sessions available — this is the baseline.' if first run."
        )
    )
    key_connections: List[str] = Field(
        description=(
            "2-3 specific connections between internal data and external research. "
            "Each must cite actual numbers from the findings. "
            "Example: 'Our email engagement of 88% exceeds the industry benchmark of 8-12%'"
        )
    )
    recommendations: List[str] = Field(
        description=(
            "3-5 actionable recommendations, each starting with a verb. "
            "Ground each in the data or research above. "
            "Example: 'Reallocate 20% of Display budget to Email — highest ROI channel.'"
        )
    )
    overall_confidence: float = Field(
        description=(
            "Overall confidence in this report's quality from 0.0 to 1.0. "
            "Lower if data was incomplete or agents had quality issues."
        ),
        ge=0.0,
        le=1.0,
    )
    data_sources: List[str] = Field(
        description=(
            "List of data sources actually used. Choose from: "
            "'Internal CSV Data', 'Web Research', 'Historical Memory (RAG)'"
        )
    )


def format_report_to_markdown(report: MarketingReport, state: MarketingState) -> str:
    """
    Converts a validated MarketingReport Pydantic object into formatted Markdown.

    This is called after successful structured output — we control the layout
    completely, so the output is always consistently formatted.
    """
    critic_flags = state.get("critic_flags", [])
    coding_conf = state.get("coding_confidence", 1.0)
    research_conf = state.get("research_confidence", 1.0)

    lines = ["# 📊 Marketing Intelligence Report\n"]

    # ── Critic warnings (if any quality flags exist) ───────────
    if critic_flags:
        lines.append("---")
        lines.append("### ⚠️ Quality Notes")
        lines.append("*The following issues were flagged during analysis:*")
        for flag in critic_flags:
            lines.append(f"- {flag}")
        lines.append("\n---")

    # ── Confidence bar ─────────────────────────────────────────
    conf = report.overall_confidence
    bar_filled = int(conf * 10)
    confidence_bar = "█" * bar_filled + "░" * (10 - bar_filled)
    conf_color = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.6 else "🔴"
    lines.append(f"**Report Confidence:** {conf_color} `{confidence_bar}` {conf:.0%}  ")
    lines.append(f"**Sources Used:** {' · '.join(report.data_sources)}\n")

    # ── Main sections ──────────────────────────────────────────
    lines.append(f"## 🎯 Original Request\n{report.original_request}")

    lines.append("## 🗺️ Analysis Performed")
    for task in report.analysis_performed:
        lines.append(f"- {task}")

    lines.append(f"\n## 📈 Internal Data Findings")
    coding_badge = f" *(confidence: {coding_conf:.0%})*" if coding_conf < 0.8 else ""
    lines.append(f"{report.internal_findings}{coding_badge}")

    lines.append(f"\n## 🌐 External Research Insights")
    research_badge = f" *(confidence: {research_conf:.0%})*" if research_conf < 0.8 else ""
    lines.append(f"{report.external_insights}{research_badge}")

    lines.append(f"\n## 📅 Historical Trends & Comparisons\n{report.historical_trends}")

    lines.append("\n## 🔗 Key Connections")
    for conn in report.key_connections:
        lines.append(f"- {conn}")

    lines.append("\n## ✅ Actionable Recommendations")
    for i, rec in enumerate(report.recommendations, 1):
        lines.append(f"**{i}.** {rec}")

    lines.append(
        f"\n---\n"
        f"*Generated by Multi-Agent BI Assistant (Phase 3 — HITL + Critic + RAG)*  \n"
        f"*Coding Quality: {coding_conf:.0%} · Research Quality: {research_conf:.0%}*"
    )

    return "\n\n".join(lines)


# ──────────────────────────────────────────────────────────────
# Fallback system prompt (used if structured output fails)
# ──────────────────────────────────────────────────────────────
FALLBACK_SYSTEM_PROMPT = """You are a senior marketing analyst. Write a professional intelligence report
using exactly these headers (do not skip any):

# 📊 Marketing Intelligence Report
## 🎯 Original Request
## 🗺️ Analysis Performed
## 📈 Internal Data Findings
## 🌐 External Research Insights
## 📅 Historical Trends & Comparisons
## 🔗 Key Connections
## ✅ Actionable Recommendations

Rules: Never hallucinate numbers. Bold key metrics. Use ⚠️ for warnings."""


def compiler_agent(state: MarketingState) -> dict:
    """
    Compiler Agent Node (Phase 3).

    Reads:  all agent outputs + memory + rag + critic_flags
    Writes: state["final_report"]
    """
    print("\n📝 [COMPILER] Synthesizing final report with structured output...")

    llm = get_llm(temperature=0.1)

    plan_text = "\n".join(
        f"  - [{'✅' if t['source'] in ('INTERNAL','BOTH') else '🌐'}] {t['task']}"
        for t in state.get("plan", [])
    )

    coding_out   = state.get("coding_output",   "No internal data analysis performed.")
    research_out = state.get("research_output", "No external research performed.")
    memory_ctx   = state.get("memory_context",  "No past sessions available.")
    rag_ctx      = state.get("rag_context",     "No historical data available.")
    critic_flags = state.get("critic_flags",    [])
    coding_conf  = state.get("coding_confidence",  1.0)
    research_conf= state.get("research_confidence",1.0)

    flags_note = ""
    if critic_flags:
        flags_note = (
            "\n\nQUALITY FLAGS FROM CRITIC (be transparent about these in the report):\n"
            + "\n".join(f"- {f}" for f in critic_flags)
        )

    user_message = (
        f"Original Query: {state['query']}\n\n"
        f"Sub-tasks completed:\n{plan_text}\n\n"
        f"--- INTERNAL DATA ---\n{coding_out}\n"
        f"(Coding confidence: {coding_conf:.0%})\n\n"
        f"--- WEB RESEARCH ---\n{research_out}\n"
        f"(Research confidence: {research_conf:.0%})\n\n"
        f"--- MEMORY CONTEXT ---\n{memory_ctx}\n\n"
        f"--- RAG HISTORICAL DATA ---\n{rag_ctx}"
        f"{flags_note}"
    )

    # ── ★ Try Pydantic structured output first ─────────────────
    try:
        structured_llm = llm.with_structured_output(MarketingReport)
        report_obj: MarketingReport = structured_llm.invoke([
            SystemMessage(content=(
                "You are a senior marketing analyst. Fill every field accurately. "
                "Use ONLY numbers and facts present in the provided data. "
                "For confidence score: start at 1.0, subtract 0.1 for each quality flag, "
                "subtract 0.15 for each skipped data source."
            )),
            HumanMessage(content=user_message),
        ])
        final_report = format_report_to_markdown(report_obj, state)
        print("   ✅ Structured report compiled successfully.")

    except Exception as e:
        # ── Fallback: unstructured Markdown (Phase 2 approach) ─
        print(f"   ⚠️  Structured output failed ({e}) — using Markdown fallback.")
        response = llm.invoke([
            SystemMessage(content=FALLBACK_SYSTEM_PROMPT),
            HumanMessage(content=user_message),
        ])
        final_report = response.content
        print("   ✅ Fallback report compiled.")

    return {"final_report": final_report}