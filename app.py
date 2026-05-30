# =============================================================
# app.py — Streamlit Frontend (Phase 2 — updated)
# =============================================================
# WHAT CHANGED FROM PHASE 1:
#   ★ New sidebar section: Memory Store (session history + clear button)
#   ★ New tab: 🧩 Memory & RAG (shows retrieved context)
#   ★ First-run warning about sentence-transformer model download
#   ★ STEPS updated for 7 nodes (was 4)
#   ★ Event handlers for memory_agent, rag_agent, save_session
#   ★ Session count displayed in sidebar
#   ★ Initial state includes memory_context + rag_context fields
# =============================================================

import os
import sys
import tempfile

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from graph.workflow import build_graph

# ── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="Marketing BI Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px 30px; border-radius: 12px;
        margin-bottom: 20px; color: white;
    }
    .memory-card {
        background: #1e1b4b; border: 1px solid #4338ca;
        border-radius: 8px; padding: 10px 14px; margin: 6px 0;
        font-size: 12px;
    }
    .phase-badge {
        background: #7c3aed; color: white;
        padding: 2px 8px; border-radius: 12px;
        font-size: 10px; font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1 style="margin:0">📊 Multi-Agent Marketing BI Assistant</h1>
    <p style="margin:4px 0 0 0; opacity:0.85;">
        Powered by LangGraph + LangChain + Groq (Llama 3.3-70B) · 100% Free
        &nbsp;&nbsp;<span class="phase-badge">Phase 2: Memory + RAG</span>
    </p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:

    # ── File Upload ───────────────────────────────────────────
    st.header("📁 Your Marketing Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file", type=["csv"],
        help="Upload your marketing data. Use data/sample_marketing_data.csv to start.",
    )

    if uploaded_file:
        if st.session_state.get("last_upload_name") != uploaded_file.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
            tmp.write(uploaded_file.getvalue())
            tmp.flush()
            st.session_state["data_path"] = tmp.name
            st.session_state["last_upload_name"] = uploaded_file.name

        st.success(f"✅ Loaded: **{uploaded_file.name}**")
        df_preview = pd.read_csv(st.session_state["data_path"])
        st.dataframe(df_preview, use_container_width=True, height=180)
        st.caption(f"📐 {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")
    else:
        st.session_state["data_path"] = None
        st.info("No data uploaded — web research only.\n\n💡 Try `data/sample_marketing_data.csv`")

    st.divider()

    # ── ★ Phase 2: Memory Store Section ───────────────────────
    st.header("🧠 Memory Store")

    # Lazy-import to avoid slow model download blocking the entire sidebar
    try:
        from memory import store as memory_store

        count = memory_store.session_count()
        st.metric("Sessions Stored", count, help="Past sessions available for retrieval")

        if count > 0:
            recent = memory_store.get_recent_sessions(limit=5)
            if recent:
                st.caption("**Recent sessions:**")
                for session in recent:
                    ts = session.get("timestamp", "")[:16].replace("T", " ")
                    q = session.get("query", "Unknown")[:50]
                    has_int = "📊" if session.get("has_internal") else ""
                    has_res = "🌐" if session.get("has_research") else ""
                    st.markdown(
                        f'<div class="memory-card">'
                        f'<b>{ts}</b><br>{q}{"..." if len(session.get("query",""))>50 else ""}<br>'
                        f'{has_int} {has_res}</div>',
                        unsafe_allow_html=True,
                    )

            st.caption("")
            if st.button("🗑️ Clear All Memory", type="secondary", use_container_width=True):
                deleted = memory_store.clear_all_sessions()
                st.success(f"Cleared {deleted} session(s).")
                st.rerun()
        else:
            st.caption("Empty — run a query to start building memory.")

        # First-run warning
        if count == 0:
            st.info(
                "⚠️ **First run?**\n\n"
                "The sentence-transformer embedding model (~90MB) "
                "will download automatically on first use. "
                "This happens once — subsequent runs are instant."
            )

    except ImportError:
        st.warning("Install Phase 2 packages:\n`pip install chromadb sentence-transformers`")

    st.divider()

    # ── Pipeline Info ─────────────────────────────────────────
    st.markdown("### 🔧 Phase 2 Pipeline")
    st.markdown("""
    1. 🧩 **Memory Agent** — retrieves past context
    2. 📋 **Planner** — informed by memory
    3. 🐍 **Coding Agent** — analyzes CSV
    4. 🔍 **Research Agent** — searches web
    5. 📁 **RAG Agent** — historical comparison
    6. 📝 **Compiler** — synthesizes everything
    7. 💾 **Save Session** — stores to ChromaDB
    """)

# ── Main: Query Input ──────────────────────────────────────────
st.subheader("💬 Ask Your Marketing Question")

with st.expander("📌 Example questions", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**📊 Data Analysis**")
        st.markdown("- *Which channel has poorest performance?*\n- *Revenue breakdown by channel?*")
    with c2:
        st.markdown("**🌐 Research Only**")
        st.markdown("- *How does attribution modeling work?*\n- *Best email marketing strategies?*")
    with c3:
        st.markdown("**🔄 Mixed (Best results)**")
        st.markdown("- *Which channel is worst and how to improve it?*\n- *Is our email performing well vs industry?*")

query = st.text_area(
    "Your question:",
    placeholder="e.g. Which marketing channel is performing the worst, and what strategies can I use to improve it?",
    height=100, key="query_input",
)

run_col, info_col = st.columns([3, 1])
with run_col:
    run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)
with info_col:
    try:
        from memory import store as _s
        _cnt = _s.session_count()
        if _cnt > 0:
            st.info(f"📚 {_cnt} session(s) in memory")
    except Exception:
        pass

# ── Pipeline Execution ─────────────────────────────────────────
if run_button:
    if not query.strip():
        st.warning("⚠️ Please enter a question before running.")
        st.stop()

    from dotenv import load_dotenv
    load_dotenv()
    if not os.getenv("GROQ_API_KEY") or os.getenv("GROQ_API_KEY") == "your_groq_api_key_here":
        st.error(
            "❌ **GROQ_API_KEY not set!**\n\n"
            "1. Go to [console.groq.com](https://console.groq.com)\n"
            "2. Create a free account → generate API key\n"
            "3. Add to `.env`: `GROQ_API_KEY=gsk_your_key_here`\n"
            "4. Restart Streamlit"
        )
        st.stop()

    graph = build_graph()

    # ★ Phase 2: initial state includes new memory/rag fields
    initial_state = {
        "query":           query.strip(),
        "data_path":       st.session_state.get("data_path"),
        "memory_context":  "",   # ★ filled by memory_agent
        "rag_context":     "",   # ★ filled by rag_agent
        "plan":            [],
        "internal_tasks":  [],
        "external_tasks":  [],
        "coding_output":   "",
        "research_output": "",
        "final_report":    "",
        "error":           None,
    }

    st.divider()
    st.subheader("⚙️ Pipeline Running...")

    progress_bar = st.progress(0, text="Starting Phase 2 pipeline...")
    status_area  = st.empty()

    # ★ 5 tabs now (added Memory & RAG)
    tab_plan, tab_memory, tab_data, tab_research, tab_report = st.tabs([
        "📋 Plan", "🧩 Memory & RAG", "🐍 Data Analysis", "🌐 Research", "📄 Final Report"
    ])

    # ★ 7 nodes now (was 4)
    STEPS = {
        "memory_agent":   1,
        "planner":        2,
        "coding_agent":   3,
        "research_agent": 4,
        "rag_agent":      5,
        "compiler":       6,
        "save_session":   7,
    }
    TOTAL = 7

    try:
        for event in graph.stream(initial_state):
            node_name   = list(event.keys())[0]
            node_output = event[node_name]
            pct = int((STEPS.get(node_name, 1) / TOTAL) * 100)

            # ── memory_agent finished ──────────────────────────
            if node_name == "memory_agent":
                progress_bar.progress(pct, text="🧩 Memory retrieved...")
                mem_ctx = node_output.get("memory_context", "")
                if mem_ctx:
                    status_area.success("✅ Relevant past sessions found!")
                else:
                    status_area.info("ℹ️ No past sessions — starting fresh.")

                with tab_memory:
                    st.subheader("🧩 Memory Agent Output")
                    if mem_ctx:
                        st.success("Past context retrieved and injected into Planner:")
                        st.markdown(mem_ctx)
                    else:
                        st.info(
                            "No relevant past sessions found in memory store.\n\n"
                            "After this run completes, a new session will be saved automatically. "
                            "Run a similar query next time to see memory in action!"
                        )

            # ── planner finished ───────────────────────────────
            elif node_name == "planner":
                progress_bar.progress(pct, text="🧠 Plan created...")
                plan = node_output.get("plan", [])
                status_area.success(f"✅ Plan ready: {len(plan)} sub-tasks")

                with tab_plan:
                    st.subheader("🗺️ Execution Plan")
                    for task in plan:
                        source = task.get("source", "?")
                        icon = {"INTERNAL": "📊", "EXTERNAL": "🌐", "BOTH": "🔄"}.get(source, "❓")
                        st.markdown(
                            f"{icon} **Task {task['id']}:** {task['task']}  \n"
                            f"&nbsp;&nbsp;&nbsp;`Source: {source}`"
                        )

            # ── coding_agent finished ──────────────────────────
            elif node_name == "coding_agent":
                progress_bar.progress(pct, text="🐍 Data analysis complete...")
                status_area.success("✅ Internal data analysis done")
                with tab_data:
                    out = node_output.get("coding_output", "")
                    if "No internal" in out or "No CSV" in out:
                        st.info("ℹ️ " + out)
                    elif "❌" in out or "error" in out.lower():
                        st.error(out)
                    else:
                        st.markdown(out)

            # ── research_agent finished ────────────────────────
            elif node_name == "research_agent":
                progress_bar.progress(pct, text="🔍 Web research complete...")
                status_area.success("✅ Market research done")
                with tab_research:
                    out = node_output.get("research_output", "")
                    if "No external research" in out:
                        st.info("ℹ️ " + out)
                    else:
                        st.markdown(out)

            # ── rag_agent finished ─────────────────────────────
            elif node_name == "rag_agent":
                progress_bar.progress(pct, text="📁 Historical data retrieved...")
                rag_ctx = node_output.get("rag_context", "")
                if "No historical" in rag_ctx or "No closely" in rag_ctx:
                    status_area.info("ℹ️ No historical comparison data yet.")
                else:
                    status_area.success("✅ Historical comparison data retrieved!")

                with tab_memory:
                    st.divider()
                    st.subheader("📁 RAG Agent Output")
                    if rag_ctx and "No historical" not in rag_ctx and "No closely" not in rag_ctx:
                        st.success("Historical comparison data retrieved for compiler:")
                        st.markdown(rag_ctx)
                    else:
                        st.info(
                            "No closely matching historical session found for comparison.\n\n"
                            "The RAG agent needs at least 1-2 prior sessions on similar topics "
                            "before it can surface meaningful trend comparisons."
                        )

            # ── compiler finished ──────────────────────────────
            elif node_name == "compiler":
                progress_bar.progress(pct, text="📝 Report compiled...")
                status_area.success("✅ Report ready!")
                with tab_report:
                    final_report = node_output.get("final_report", "")
                    if final_report:
                        st.markdown(final_report)
                        st.divider()
                        st.download_button(
                            label="⬇️ Download Report as Markdown",
                            data=final_report,
                            file_name="marketing_intelligence_report.md",
                            mime="text/markdown",
                            use_container_width=True,
                        )
                    else:
                        st.warning("Compiler returned an empty report.")

            # ── save_session finished ──────────────────────────
            elif node_name == "save_session":
                progress_bar.progress(100, text="💾 Session saved to memory!")
                status_area.success("🎉 Complete — session saved to memory for future use.")

        st.success(
            "🎉 Analysis complete! Click **📄 Final Report** to read your report. "
            "This session has been saved — future similar queries will reference it."
        )
        # Prompt rerun to refresh the memory sidebar count
        st.rerun()

    except Exception as e:
        progress_bar.progress(0, text="Error")
        st.error(f"❌ Pipeline failed: {str(e)}")
        with st.expander("🔍 Full error details"):
            st.exception(e)