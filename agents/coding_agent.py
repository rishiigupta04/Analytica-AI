# =============================================================
# agents/coding_agent.py — The Coding Agent
# =============================================================
# ROLE: Data analyst for internal CSV data.
#
# WHAT IT DOES:
#   1. Receives internal_tasks from state (set by Planner)
#   2. Loads the uploaded CSV into a Pandas DataFrame
#   3. Shows the LLM the schema + sample rows
#   4. Asks the LLM to write Python/Pandas code to answer the tasks
#   5. Executes that code safely
#   6. Returns a formatted string of findings
#
# HOW CODE EXECUTION WORKS:
#   We use Python's built-in exec() to run the LLM-generated code.
#   The LLM is instructed to store its findings in a variable
#   called `analysis_results`. We then read that variable back.
#
#   This is simpler and more reliable than a full ReAct agent
#   for structured data analysis tasks.
# =============================================================

import numpy as np
import pandas as pd
from langchain_core.messages import HumanMessage, SystemMessage

from state import MarketingState
from utils.llm import get_llm


def _to_markdown_table(rows: list[dict]) -> str:
    table_df = pd.DataFrame(rows)
    if table_df.empty:
        return "| Finding |\n| --- |\n| No analysis rows returned |"

    columns = list(table_df.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    body = []
    for _, row in table_df.iterrows():
        body.append("| " + " | ".join(str(row[col]) for col in columns) + " |")
    return "\n".join([header, separator, *body])


def coding_agent(state: MarketingState) -> dict:
    """
    Coding Agent Node.

    Reads:  state["internal_tasks"], state["data_path"]
    Writes: state["coding_output"]
    """

    # ── Guard: skip if no internal work to do ─────────────────
    if not state.get("internal_tasks"):
        print("   ⏭️  [CODING AGENT] No internal tasks — skipping.")
        return {
            "coding_output": "No internal data analysis required for this query.",
            "coding_summary": "No internal data analysis was needed for this query.",
            "coding_table_rows": [],
        }

    if not state.get("data_path"):
        print("   ⏭️  [CODING AGENT] No data file — skipping.")
        return {
            "coding_output": "No CSV file was uploaded, so internal analysis was skipped.",
            "coding_summary": "No CSV file was uploaded, so internal analysis could not be performed.",
            "coding_table_rows": [],
        }

    print(f"\n🐍 [CODING AGENT] Analyzing {len(state['internal_tasks'])} task(s) from CSV...")

    # ── Step 1: Load the CSV into a DataFrame ─────────────────
    try:
        df = pd.read_csv(state["data_path"])
    except Exception as e:
        error_message = f"❌ Failed to read CSV file: {str(e)}"
        return {
            "coding_output": error_message,
            "coding_summary": error_message,
            "coding_table_rows": [{"Issue": "Failed to read CSV file", "Details": str(e)}],
        }

    # ── Step 2: Build a schema description for the LLM ────────
    # We give the LLM a "map" of the data so it knows what columns exist
    schema_info = (
        f"Shape: {df.shape[0]} rows × {df.shape[1]} columns\n"
        f"Columns and types:\n{df.dtypes.to_string()}\n\n"
        f"First 3 rows:\n{df.head(3).to_string(index=False)}\n\n"
        f"Basic statistics:\n{df.describe().to_string()}"
    )

    tasks_text = "\n".join(
        f"Task {t['id']}: {t['task']}" for t in state["internal_tasks"]
    )

    # ── Step 3: Ask LLM to write Python code ──────────────────
    llm = get_llm(temperature=0)

    code_prompt = f"""You are a senior Python data analyst. The pandas DataFrame is already loaded as `df`.

DATA SCHEMA:
{schema_info}

TASKS TO COMPLETE:
{tasks_text}

Write Python code that:
1. Uses pandas to analyze `df` and answer ALL tasks above
2. Stores the findings in a variable called `analysis_results` as a dictionary with:
   - "summary": a short paragraph summary of the table
   - "table_rows": a list of dictionaries, one per row in the analysis table
3. Uses f-strings to embed actual data values (channel names, numbers, percentages)
4. Ensures the table rows are clear, specific, and directly answer the tasks
5. Handles missing or unexpected columns gracefully with a clear fallback summary and minimal table rows
6. Prefers precise numeric formatting (percentages, currency) and avoids rounding away meaning
7. Does NOT use print() — only assign to `analysis_results`

Return ONLY executable Python code. No explanations. No markdown fences.

Example of good output format:
worst = df.nsmallest(1, 'Revenue')
analysis_results = {{
    "summary": (
        f"Email is the weakest channel by revenue at ${{worst['Revenue'].values[0]:,.0f}}, "
        f"contributing only {{(worst['Revenue'].values[0]/df['Revenue'].sum()*100):.1f}}% of total revenue."
    ),
    "table_rows": [
        {{
            "Task": "Worst channel by revenue",
            "Channel": worst["Channel"].values[0],
            "Revenue": f"${{worst['Revenue'].values[0]:,.0f}}",
            "Share of Total Revenue": f"{{(worst['Revenue'].values[0]/df['Revenue'].sum()*100):.1f}}%",
        }}
    ],
}}
"""

    print("   🤖 Generating analysis code...")
    response = llm.invoke([HumanMessage(content=code_prompt)])

    # ── Step 4: Clean up the generated code ───────────────────
    code = response.content.strip()
    if "```python" in code:
        code = code.split("```python")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()

    print("   ⚙️  Executing generated code...")

    # ── Step 5: Execute the code safely ───────────────────────
    # We pass `df`, `pd`, and `np` as available variables
    # The LLM's code will write into `exec_env["analysis_results"]`
    exec_env = {
        "df": df,
        "pd": pd,
        "np": np,
    }

    try:
        exec(code, exec_env)  # noqa: S102
        analysis_results = exec_env.get("analysis_results")
        if analysis_results is None:
            analysis_results = "⚠️ Code ran but did not produce `analysis_results`."
        print("   ✅ Code executed successfully.")

    except Exception as exec_error:
        # If execution fails, return a helpful error + the attempted code
        analysis_results = (
            f"⚠️ Code execution encountered an error: {exec_error}\n\n"
            f"Attempted code:\n{code}"
        )
        print(f"   ❌ Execution error: {exec_error}")

    if isinstance(analysis_results, dict):
        summary = str(analysis_results.get("summary", "")).strip()
        table_rows = analysis_results.get("table_rows", [])
        if not isinstance(table_rows, list):
            table_rows = []
    else:
        summary = str(analysis_results).strip()
        table_rows = [{"Finding": summary}] if summary else []

    if not summary:
        summary = "The analysis results are shown in the table below."

    if not table_rows:
        table_rows = [{"Finding": summary}]

    table_markdown = _to_markdown_table(table_rows)

    return {
        "coding_output": f"=== 📊 Internal Data Analysis ===\n\n{summary}\n\n{table_markdown}",
        "coding_summary": summary,
        "coding_table_rows": table_rows,
    }
