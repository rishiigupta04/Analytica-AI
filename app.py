# =============================================================
# app.py — Streamlit Frontend (Phase 3 — complete rewrite)
# =============================================================
# WHAT CHANGED FROM PHASE 2:
#   ★ Two-phase streaming: pre-HITL → interrupt → post-HITL
#   ★ Plan editing UI: edit task text, source tags, remove tasks
#   ★ Graph cached in session_state so MemorySaver persists
#     across Streamlit re-runs (critical for interrupt/resume)
#   ★ Critic confidence badges in new Critic tab
#   ★ 7-step progress bar (was 7 in Phase 2, now 10 with critics)
#   ★ Clear visual separation of HITL approval vs pipeline states
#
# STREAMLIT + HITL FLOW:
#   The core challenge: Streamlit re-runs the entire script on
#   every user interaction (button click, file upload, etc).
#   But graph.stream() is a generator that needs to be called
#   across two separate Streamlit runs (pre and post HITL).
#
#   Solution:
#     1. Cache the compiled graph in st.session_state
#        → same MemorySaver instance survives re-runs
#     2. Store thread_id in st.session_state
#        → same graph checkpoint identified across re-runs
#     3. st.session_state["pipeline_phase"] controls which
#        UI section renders on each Streamlit re-run:
#          "idle"          → show query input
#          "pre_hitl"      → streaming memory_agent + planner
#          "hitl_waiting"  → show plan approval UI
#          "post_hitl"     → streaming remaining agents
#          "done"          → show final report
# =============================================================

import os
import sys
import uuid
import tempfile

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from langgraph.types import Command

# ── Page config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing BI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .main-header {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
    padding: 22px 30px; border-radius: 14px; margin-bottom: 20px; color: white;
  }
  .hitl-box {
    background: #1a1a2e; border: 2px solid #f59e0b;
    border-radius: 12px; padding: 20px; margin: 14px 0;
  }
  .task-card {
    background: #0f172a; border: 1px solid #334155;
    border-radius: 8px; padding: 14px; margin: 8px 0;
  }
  .critic-pass { color: #4ade80; font-weight: 700; }
  .critic-fail { color: #f87171; font-weight: 700; }
  .phase-badge {
    background: #7c3aed; color: white; padding: 2px 9px;
    border-radius: 12px; font-size: 10px; font-weight: 700;
  }
  .memory-card {
    background: #1e1b4b; border: 1px solid #4338ca;
    border-radius: 8px; padding: 10px 14px; margin: 6px 0; font-size: 12px;
  }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
  <h1 style="margin:0">📊 Multi-Agent Marketing BI Assistant</h1>
  <p style="margin:5px 0 0 0; opacity:0.88;">
    LangGraph · LangChain · Groq Llama 3.3-70B · 100% Free Stack
    &nbsp;<span class="phase-badge">Phase 3: HITL + Critic + Pydantic</span>
  </p>
</div>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# Called once per browser session; sets safe defaults.
# ──────────────────────────────────────────────────────────────
def init_session_state():
    defaults = {
        "pipeline_phase":   "idle",   # idle | pre_hitl | hitl_waiting | post_hitl | done
        "thread_id":        None,
        "graph":            None,     # cached compiled graph (keeps MemorySaver alive)
        "data_path":        None,
        "last_upload_name": None,
        "hitl_data":        None,     # interrupt payload from graph
        "pre_hitl_events":  {},       # tab content collected before interrupt
        "all_tab_data":     {},       # all tab content accumulated across both phases
        "query":            "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session_state()


# ──────────────────────────────────────────────────────────────
# GRAPH FACTORY (cached in session_state)
# ──────────────────────────────────────────────────────────────
def get_graph():
    """
    Returns the cached compiled graph, building it once per session.
    CRITICAL: must be cached so the same MemorySaver instance is
    reused across Streamlit re-runs — otherwise the HITL checkpoint is lost.
    """
    if st.session_state["graph"] is None:
        from graph.workflow import build_graph
        st.session_state["graph"] = build_graph()
    return st.session_state["graph"]


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:

    # ── File Upload ───────────────────────────────────────────
    st.header("📁 Your Marketing Data")
    uploaded_file = st.file_uploader(
        "Upload CSV", type=["csv"],
        help="Upload your channel performance data. Try data/sample_marketing_data.csv",
        disabled=(st.session_state["pipeline_phase"] not in ("idle", "done")),
    )

    if uploaded_file:
        if st.session_state["last_upload_name"] != uploaded_file.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uploaded_file.getvalue())
            tmp.flush()
            st.session_state["data_path"] = tmp.name
            st.session_state["last_upload_name"] = uploaded_file.name
        st.success(f"✅ **{uploaded_file.name}**")
        df_p = pd.read_csv(st.session_state["data_path"])
        st.dataframe(df_p, use_container_width=True, height=170)
        st.caption(f"📐 {df_p.shape[0]} rows × {df_p.shape[1]} cols")
    else:
        if st.session_state["pipeline_phase"] == "idle":
            st.session_state["data_path"] = None
        st.info("No data → web research only.\n\n💡 Try `data/sample_marketing_data.csv`")

    st.divider()

    # ── Memory Store ───────────────────────────────────────────
    st.header("🧠 Memory Store")
    try:
        from memory import store as memory_store
        count = memory_store.session_count()
        st.metric("Sessions Stored", count)
        if count > 0:
            recent = memory_store.get_recent_sessions(5)
            for s in recent:
                ts = s.get("timestamp", "")[:16].replace("T", " ")
                q  = s.get("query", "")[:48]
                st.markdown(
                    f'<div class="memory-card"><b>{ts}</b><br>{q}</div>',
                    unsafe_allow_html=True,
                )
            if st.button("🗑️ Clear Memory", use_container_width=True):
                memory_store.clear_all_sessions()
                st.success("Memory cleared.")
                st.rerun()
        else:
            st.caption("Empty — run a query to build memory.")
    except ImportError:
        st.warning("`pip install chromadb sentence-transformers`")

    st.divider()

    # ── Pipeline Map ──────────────────────────────────────────
    st.markdown("### 🔧 Phase 3 Pipeline")
    st.markdown("""
    1. 🧩 **Memory Agent** — past context
    2. 📋 **Planner** — decompose query
    3. 🧑‍✈️ **HITL Gate** ← *you review here*
    4. 🐍 **Coding Agent** — CSV analysis
    5. 🪞 **Critic** — validate coding
    6. 🔍 **Research Agent** — web search
    7. 🪞 **Critic** — validate research
    8. 📁 **RAG Agent** — historical data
    9. 📝 **Compiler** — Pydantic report
    10. 💾 **Save Session** → ChromaDB
    """)

    # ── Reset button ──────────────────────────────────────────
    if st.session_state["pipeline_phase"] != "idle":
        st.divider()
        if st.button("🔄 New Query", use_container_width=True, type="secondary"):
            keys_to_reset = [
                "pipeline_phase", "thread_id", "hitl_data",
                "pre_hitl_events", "all_tab_data", "query",
            ]
            for k in keys_to_reset:
                st.session_state[k] = None if k in ("thread_id", "hitl_data") else (
                    {} if k in ("pre_hitl_events", "all_tab_data") else
                    "idle" if k == "pipeline_phase" else ""
                )
            st.rerun()


# ──────────────────────────────────────────────────────────────
# HELPER: event processor — converts streamed events into tab content
# ──────────────────────────────────────────────────────────────
def process_event(event: dict, tabs_data: dict, progress_bar, status_area, step: int, total: int):
    """
    Process one stream event and store rendered content in tabs_data.
    tabs_data keys: "plan", "memory", "data", "research", "critic", "report"
    """
    node_name   = list(event.keys())[0]
    node_output = event[node_name]
    pct = int((step / total) * 100)

    if node_name == "memory_agent":
        progress_bar.progress(pct, text="🧩 Memory agent complete...")
        mem = node_output.get("memory_context", "")
        tabs_data.setdefault("memory", [])
        tabs_data["memory"].append(("memory", mem))
        status_area.info(f"🧩 Memory: {'Context found ✅' if mem else 'No past sessions'}")

    elif node_name == "planner":
        progress_bar.progress(pct, text="🧠 Plan created — awaiting your approval...")
        plan = node_output.get("plan", [])
        tabs_data.setdefault("plan", [])
        tabs_data["plan"].append(("plan", plan))
        status_area.warning(f"🧑‍✈️ Plan ready ({len(plan)} tasks) — review before continuing")

    elif node_name == "hitl_node":
        progress_bar.progress(pct, text="✅ Plan approved — pipeline resuming...")
        status_area.success("✅ Plan approved — running agents...")

    elif node_name == "coding_agent":
        progress_bar.progress(pct, text="🐍 Data analysis running...")
        out = node_output.get("coding_output", "")
        retries = node_output.get("coding_retry_count", 1)
        tabs_data.setdefault("data", [])
        tabs_data["data"].append(("coding", out, retries))
        status_area.info(f"🐍 Data analysis complete (attempt {retries})")

    elif node_name == "critique_coding":
        conf = node_output.get("coding_confidence", 1.0)
        flags = node_output.get("critic_flags", [])
        tabs_data.setdefault("critic", [])
        tabs_data["critic"].append(("coding_critique", conf, flags))
        icon = "✅" if conf >= 0.6 else "🔄"
        status_area.info(f"{icon} Coding critique: {conf:.0%} confidence")

    elif node_name == "research_agent":
        progress_bar.progress(pct, text="🔍 Web research running...")
        out = node_output.get("research_output", "")
        retries = node_output.get("research_retry_count", 1)
        tabs_data.setdefault("research", [])
        tabs_data["research"].append(("research", out, retries))
        status_area.info(f"🔍 Research complete (attempt {retries})")

    elif node_name == "critique_research":
        conf = node_output.get("research_confidence", 1.0)
        flags = node_output.get("critic_flags", [])
        tabs_data.setdefault("critic", [])
        tabs_data["critic"].append(("research_critique", conf, flags))
        icon = "✅" if conf >= 0.6 else "🔄"
        status_area.info(f"{icon} Research critique: {conf:.0%} confidence")

    elif node_name == "rag_agent":
        progress_bar.progress(pct, text="📁 Historical data retrieved...")
        rag = node_output.get("rag_context", "")
        tabs_data.setdefault("memory", [])
        tabs_data["memory"].append(("rag", rag))
        status_area.info("📁 RAG agent complete")

    elif node_name == "compiler":
        progress_bar.progress(pct, text="📝 Compiling final report...")
        report = node_output.get("final_report", "")
        tabs_data["report"] = report
        status_area.success("📝 Report compiled!")

    elif node_name == "save_session":
        progress_bar.progress(100, text="💾 Session saved to memory!")
        status_area.success("🎉 Complete — session saved for future reference.")

    return node_name


# ──────────────────────────────────────────────────────────────
# HITL PLAN EDITING UI
# ──────────────────────────────────────────────────────────────
def render_hitl_approval_ui():
    """
    Renders the plan approval interface.
    User can edit task descriptions, change source tags, remove tasks.
    Returns the final plan when user clicks approve.
    """
    hitl_data = st.session_state["hitl_data"]
    if not hitl_data:
        return

    plan = hitl_data.get("plan", [])
    query = hitl_data.get("query", "")
    has_data = hitl_data.get("has_data", False)
    memory_ctx = hitl_data.get("memory_context", "")

    st.markdown("""
    <div class="hitl-box">
      <h3 style="color:#fbbf24; margin-top:0">🧑‍✈️ Human-in-the-Loop — Plan Review Gate</h3>
      <p style="color:#cbd5e1">
        The Planner has decomposed your query into the tasks below.
        Review each task, edit descriptions or source tags if needed, then approve.
      </p>
    </div>
    """, unsafe_allow_html=True)

    # Show memory context if available
    if memory_ctx:
        with st.expander("📚 Memory context used by Planner", expanded=False):
            st.markdown(memory_ctx)

    st.subheader(f"📋 Plan for: *{query[:80]}{'...' if len(query)>80 else ''}*")
    st.caption(f"{'✅ CSV data available for INTERNAL tasks' if has_data else '⚠️ No CSV uploaded — only EXTERNAL tasks will run'}")

    # ── Editable task cards ────────────────────────────────────
    edited_tasks = []
    removed_ids  = []

    for i, task in enumerate(plan):
        with st.container():
            st.markdown(f"<div class='task-card'>", unsafe_allow_html=True)
            col_main, col_src, col_del = st.columns([5, 2, 1])

            with col_main:
                new_desc = st.text_input(
                    f"Task {task['id']}",
                    value=task["task"],
                    key=f"task_desc_{i}",
                    label_visibility="collapsed",
                )

            with col_src:
                src_options = ["INTERNAL", "EXTERNAL", "BOTH"] if has_data else ["EXTERNAL"]
                current_src = task["source"] if task["source"] in src_options else "EXTERNAL"
                new_src = st.selectbox(
                    "Source",
                    options=src_options,
                    index=src_options.index(current_src),
                    key=f"task_src_{i}",
                    label_visibility="collapsed",
                    help="INTERNAL=CSV, EXTERNAL=Web, BOTH=both",
                )

            with col_del:
                remove = st.checkbox("❌", key=f"task_remove_{i}",
                                     help="Remove this task")

            if not remove:
                edited_tasks.append({
                    "id": task["id"],
                    "task": new_desc,
                    "source": new_src,
                })
            else:
                removed_ids.append(task["id"])
            st.markdown("</div>", unsafe_allow_html=True)

    # Summary
    if removed_ids:
        st.info(f"ℹ️ Tasks marked for removal: {', '.join(removed_ids)}")
    if not edited_tasks:
        st.error("⚠️ All tasks removed — please keep at least one task.")

    st.divider()

    col_approve, col_cancel = st.columns([3, 1])
    with col_approve:
        approve_disabled = len(edited_tasks) == 0
        if st.button(
            "✅ Approve Plan & Run Pipeline",
            type="primary",
            use_container_width=True,
            disabled=approve_disabled,
        ):
            return edited_tasks  # Return approved plan

    with col_cancel:
        if st.button("↩️ Cancel", use_container_width=True):
            st.session_state["pipeline_phase"] = "idle"
            st.rerun()

    return None  # Not yet approved


# ──────────────────────────────────────────────────────────────
# RENDER ACCUMULATED TABS
# ──────────────────────────────────────────────────────────────
def render_tabs(tabs_data: dict):
    """Renders all accumulated tab content from both pipeline phases."""
    tab_plan, tab_memory, tab_data, tab_research, tab_critic, tab_report = st.tabs([
        "📋 Plan", "🧩 Memory & RAG", "🐍 Data Analysis",
        "🌐 Research", "🪞 Critic", "📄 Final Report"
    ])

    # Plan tab
    with tab_plan:
        for item in tabs_data.get("plan", []):
            st.subheader("🗺️ Approved Execution Plan")
            for task in item[1]:
                src = task.get("source", "?")
                icon = {"INTERNAL":"📊","EXTERNAL":"🌐","BOTH":"🔄"}.get(src,"❓")
                st.markdown(f"{icon} **Task {task['id']}:** {task['task']}  \n&nbsp;&nbsp;&nbsp;`{src}`")

    # Memory tab
    with tab_memory:
        for item in tabs_data.get("memory", []):
            if item[0] == "memory":
                st.subheader("🧩 Memory Agent")
                if item[1]:
                    st.success("Past context found and injected into Planner:")
                    st.markdown(item[1])
                else:
                    st.info("No relevant past sessions found — starting fresh.")
            elif item[0] == "rag":
                st.subheader("📁 RAG Agent")
                if item[1] and "No historical" not in item[1]:
                    st.success("Historical comparison data retrieved:")
                    st.markdown(item[1])
                else:
                    st.info("No closely matching historical session for comparison.")

    # Data tab
    with tab_data:
        for item in tabs_data.get("data", []):
            attempt = item[2]
            if attempt > 1:
                st.caption(f"📌 Showing result from attempt {attempt}/3")
            out = item[1]
            if "No internal" in out or "No CSV" in out:
                st.info("ℹ️ " + out)
            elif "❌" in out or "error" in out.lower():
                st.error(out)
            else:
                st.markdown(out)

    # Research tab
    with tab_research:
        for item in tabs_data.get("research", []):
            attempt = item[2]
            if attempt > 1:
                st.caption(f"📌 Showing result from attempt {attempt}/3")
            out = item[1]
            if "No external" in out:
                st.info("ℹ️ " + out)
            else:
                st.markdown(out)

    # Critic tab
    with tab_critic:
        critiques = tabs_data.get("critic", [])
        if not critiques:
            st.info("Critic results will appear here after agents run.")
        for item in critiques:
            kind = item[0]
            conf = item[1]
            flags = item[2]

            label = "🐍 Coding Agent Critique" if kind == "coding_critique" else "🔍 Research Agent Critique"
            emoji = "🟢" if conf >= 0.8 else "🟡" if conf >= 0.6 else "🔴"
            passed = conf >= 0.6

            st.subheader(label)
            col_conf, col_pass = st.columns(2)
            with col_conf:
                bar = "█" * int(conf*10) + "░" * (10-int(conf*10))
                st.markdown(f"**Confidence:** {emoji} `{bar}` **{conf:.0%}**")
            with col_pass:
                if passed:
                    st.markdown("<span class='critic-pass'>✅ PASSED</span>", unsafe_allow_html=True)
                else:
                    st.markdown("<span class='critic-fail'>⚠️ RETRIED</span>", unsafe_allow_html=True)

            if flags:
                with st.expander("Quality flags raised"):
                    for f in flags:
                        st.markdown(f"- {f}")
            st.divider()

    # Report tab
    with tab_report:
        report = tabs_data.get("report", "")
        if report:
            st.markdown(report)
            st.divider()
            st.download_button(
                "⬇️ Download Report (.md)",
                data=report,
                file_name="marketing_intelligence_report.md",
                mime="text/markdown",
                use_container_width=True,
            )
        else:
            st.info("Final report will appear here after compilation.")


# ──────────────────────────────────────────────────────────────
# PHASE: IDLE — Query Input Form
# ──────────────────────────────────────────────────────────────
if st.session_state["pipeline_phase"] == "idle":

    st.subheader("💬 Ask Your Marketing Question")

    with st.expander("📌 Example questions", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📊 Data Analysis**")
            st.markdown("- *Which channel performs worst?*\n- *Revenue breakdown?*")
        with c2:
            st.markdown("**🌐 Research Only**")
            st.markdown("- *How does attribution work?*\n- *Email marketing best practices?*")
        with c3:
            st.markdown("**🔄 Mixed (Best)**")
            st.markdown("- *Worst channel + improvement strategy?*\n- *Email vs industry benchmark?*")

    query = st.text_area(
        "Your question:",
        placeholder="e.g. Which marketing channel is performing worst, and how can I improve it?",
        height=100, key="query_input",
    )

    if st.button("🚀 Analyze", type="primary", use_container_width=True):
        if not query.strip():
            st.warning("Please enter a question.")
            st.stop()

        from dotenv import load_dotenv
        load_dotenv()
        if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
            st.error("❌ Set GROQ_API_KEY in .env file first. Get free key at console.groq.com")
            st.stop()

        st.session_state["query"]          = query.strip()
        st.session_state["pipeline_phase"] = "pre_hitl"
        st.session_state["thread_id"]      = str(uuid.uuid4())
        st.session_state["all_tab_data"]   = {}
        st.rerun()


# ──────────────────────────────────────────────────────────────
# PHASE: PRE-HITL — Stream memory_agent + planner, stop at interrupt
# ──────────────────────────────────────────────────────────────
elif st.session_state["pipeline_phase"] == "pre_hitl":

    st.info("🚀 Pipeline starting — running Memory Agent and Planner...")

    graph = get_graph()
    thread_id = st.session_state["thread_id"]
    config = {"configurable": {"thread_id": thread_id}}

    initial_state = {
        "query":                   st.session_state["query"],
        "data_path":               st.session_state["data_path"],
        "memory_context":          "",
        "rag_context":             "",
        "plan":                    [],
        "internal_tasks":          [],
        "external_tasks":          [],
        "plan_approved":           False,
        "critic_flags":            [],
        "coding_retry_count":      0,
        "research_retry_count":    0,
        "coding_confidence":       0.0,
        "research_confidence":     0.0,
        "critic_coding_feedback":  "",
        "critic_research_feedback": "",
        "coding_output":           "",
        "research_output":         "",
        "final_report":            "",
        "error":                   None,
    }

    progress_bar = st.progress(0, text="Starting pipeline...")
    status_area  = st.empty()
    tabs_data    = st.session_state.get("all_tab_data", {})
    step = 0

    try:
        for event in graph.stream(initial_state, config=config, stream_mode="updates"):

            if "__interrupt__" in event:
                # Graph paused at HITL node
                interrupt_obj = event["__interrupt__"][0]
                st.session_state["hitl_data"]        = interrupt_obj.value
                st.session_state["pipeline_phase"]   = "hitl_waiting"
                st.session_state["all_tab_data"]     = tabs_data
                progress_bar.progress(30, text="🧑‍✈️ Waiting for your plan approval...")
                st.rerun()
                break

            step += 1
            node_name = process_event(event, tabs_data, progress_bar, status_area, step, 10)

    except Exception as e:
        st.error(f"❌ Pre-HITL pipeline error: {e}")
        with st.expander("Full error"):
            st.exception(e)
        st.session_state["pipeline_phase"] = "idle"


# ──────────────────────────────────────────────────────────────
# PHASE: HITL_WAITING — Show plan editing UI
# ──────────────────────────────────────────────────────────────
elif st.session_state["pipeline_phase"] == "hitl_waiting":

    st.progress(30, text="🧑‍✈️ Awaiting your plan approval...")

    # Show pre-HITL tabs (plan + memory so far)
    render_tabs(st.session_state.get("all_tab_data", {}))

    st.divider()

    # Render approval UI — returns edited plan if approved
    approved_plan = render_hitl_approval_ui()

    if approved_plan is not None:
        # User clicked approve → save plan and move to post_hitl phase
        st.session_state["approved_plan"]    = approved_plan
        st.session_state["pipeline_phase"]   = "post_hitl"
        st.rerun()


# ──────────────────────────────────────────────────────────────
# PHASE: POST-HITL — Resume graph with Command(resume=...)
# ──────────────────────────────────────────────────────────────
elif st.session_state["pipeline_phase"] == "post_hitl":

    st.success("✅ Plan approved — running remaining agents...")

    graph     = get_graph()
    thread_id = st.session_state["thread_id"]
    config    = {"configurable": {"thread_id": thread_id}}

    # This is what hitl_node receives as `user_response` from interrupt()
    resume_value = {"plan": st.session_state["approved_plan"]}

    progress_bar = st.progress(35, text="🔄 Pipeline resuming after HITL approval...")
    status_area  = st.empty()
    tabs_data    = st.session_state.get("all_tab_data", {})
    step = 3  # memory(1) + planner(2) + hitl(3) already done

    # Render tabs showing pre-HITL content
    tab_containers = st.empty()

    try:
        for event in graph.stream(
            Command(resume=resume_value),
            config=config,
            stream_mode="updates",
        ):
            if "__interrupt__" in event:
                break  # Shouldn't happen in post-HITL but guard anyway

            step += 1
            node_name = process_event(event, tabs_data, progress_bar, status_area, step, 10)
            st.session_state["all_tab_data"] = tabs_data

            if node_name == "save_session":
                st.session_state["pipeline_phase"] = "done"

    except Exception as e:
        st.error(f"❌ Post-HITL pipeline error: {e}")
        with st.expander("Full error"):
            st.exception(e)

    st.rerun()


# ──────────────────────────────────────────────────────────────
# PHASE: DONE — Display final results
# ──────────────────────────────────────────────────────────────
elif st.session_state["pipeline_phase"] == "done":

    st.success(
        "🎉 **Analysis complete!** "
        "Click **📄 Final Report** tab for your report. "
        "This session has been saved to memory."
    )
    st.progress(100, text="✅ Pipeline complete")

    render_tabs(st.session_state.get("all_tab_data", {}))

    st.divider()
    if st.button("🔄 Run Another Query", type="secondary"):
        st.session_state["pipeline_phase"] = "idle"
        st.session_state["all_tab_data"]   = {}
        st.session_state["hitl_data"]      = None
        st.rerun()