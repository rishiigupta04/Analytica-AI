# =============================================================
# agents/coding_agent.py — The Coding Agent (Phase 3 — updated)
# =============================================================
# WHAT CHANGED FROM PHASE 2:
#   ★ Reads state["coding_retry_count"] to know if this is a retry
#   ★ Reads state["critic_coding_feedback"] on retries to improve
#   ★ Increments coding_retry_count in its return value
#   ★ Adds retry context to the LLM code-generation prompt
#
# RETRY BEHAVIOUR:
#   First run (retry_count=0): normal behavior from Phase 2
#   Retry run (retry_count>0): adds critic's feedback to the
#   code-generation prompt so the LLM knows what went wrong
#   and produces a better attempt.
#
#   Example critic feedback that gets added to prompt:
#   "The Revenue column is named 'Total_Revenue' — use that exact
#    column name. Also sort by both revenue AND conversion_rate."
# =============================================================

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from state import MarketingState
from utils.llm import get_llm


def coding_agent(state: MarketingState) -> dict:
    """
    Coding Agent Node (Phase 3).

    Reads:  state["internal_tasks"], state["data_path"],
            state["coding_retry_count"] ★, state["critic_coding_feedback"] ★
    Writes: state["coding_output"], state["coding_retry_count"] ★
    """
    # ── Guard ──────────────────────────────────────────────────
    if not state.get("internal_tasks"):
        return {
            "coding_output": "No internal data analysis required for this query.",
            "coding_retry_count": 0,
        }

    if not state.get("data_path"):
        return {
            "coding_output": "No CSV file was uploaded, so internal analysis was skipped.",
            "coding_retry_count": 0,
        }

    # ★ Phase 3: track retry context
    retry_count = state.get("coding_retry_count", 0)
    critic_feedback = state.get("critic_coding_feedback", "")

    attempt_label = f"attempt {retry_count + 1}/3" if retry_count > 0 else "first attempt"
    print(f"\n🐍 [CODING AGENT] {attempt_label} — "
          f"{len(state['internal_tasks'])} task(s)")

    # ── Load CSV ───────────────────────────────────────────────
    try:
        df = pd.read_csv(state["data_path"])
    except Exception as e:
        return {
            "coding_output": f"❌ Failed to read CSV: {str(e)}",
            "coding_retry_count": retry_count + 1,
        }

    # ── Build schema description ───────────────────────────────
    schema_info = (
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Columns and types:\n{df.dtypes.to_string()}\n\n"
        f"First 3 rows:\n{df.head(3).to_string(index=False)}\n\n"
        f"Statistics:\n{df.describe().to_string()}"
    )

    tasks_text = "\n".join(
        f"Task {t['id']}: {t['task']}" for t in state["internal_tasks"]
    )

    # ★ Phase 3: add critic feedback on retry
    retry_context = ""
    if retry_count > 0 and critic_feedback:
        retry_context = (
            f"\n\n⚠️ RETRY IMPROVEMENT REQUIRED:\n"
            f"Your previous attempt was rejected by the quality reviewer.\n"
            f"Specific feedback to address: {critic_feedback}\n"
            f"Make sure to fix all mentioned issues in this attempt."
        )

    llm = get_llm(temperature=0)

    code_prompt = (
        f"You are a Python data analyst. The pandas DataFrame is already loaded as `df`.\n\n"
        f"DATA SCHEMA:\n{schema_info}\n\n"
        f"TASKS TO COMPLETE:\n{tasks_text}"
        f"{retry_context}\n\n"
        f"Write Python code that:\n"
        f"1. Uses pandas to analyze `df` and answer ALL tasks above\n"
        f"2. Stores COMPLETE findings as a formatted string in `analysis_results`\n"
        f"3. Embeds ACTUAL DATA VALUES using f-strings (channel names, numbers, %)\n"
        f"4. Does NOT use print() — only assign to `analysis_results`\n"
        f"5. Uses EXACT column names from the schema above\n\n"
        f"Return ONLY executable Python code. No explanations. No markdown fences."
    )

    print("   🤖 Generating analysis code...")
    response = llm.invoke([HumanMessage(content=code_prompt)])

    code = response.content.strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    print("   ⚙️  Executing generated code...")
    exec_env = {"df": df, "pd": pd, "np": np}

    try:
        exec(code, exec_env)  # noqa: S102
        analysis_results = exec_env.get(
            "analysis_results",
            "⚠️ Code ran but did not produce `analysis_results`."
        )
        print("   ✅ Code executed successfully.")
    except Exception as exec_error:
        analysis_results = (
            f"⚠️ Code execution error: {exec_error}\n\nAttempted code:\n{code}"
        )
        print(f"   ❌ Execution error: {exec_error}")

    return {
        "coding_output": f"=== 📊 Internal Data Analysis ===\n\n{analysis_results}",
        "coding_retry_count": retry_count + 1,   # ★ increment each run
    }