"""FastAPI wrapper for the LangGraph pipeline.

This is an optional backend for the React frontend and does not replace Streamlit.
"""

from __future__ import annotations

import json
import os
import tempfile
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from graph.workflow import build_graph

try:
    from memory import store as memory_store
except Exception:
    memory_store = None


app = FastAPI(title="Marketing BI Assistant API", version="0.1.0")

allowed_origins = os.getenv("ALLOWED_ORIGINS", "*")
origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins or ["*"],
    allow_credentials=True,
    allow_methods=["*"] ,
    allow_headers=["*"],
)


def _ensure_api_key() -> None:
    load_dotenv()
    key = os.getenv("GROQ_API_KEY")
    if not key or key == "your_groq_api_key_here":
        raise HTTPException(
            status_code=400,
            detail=(
                "GROQ_API_KEY not set. Add GROQ_API_KEY to .env and restart."
            ),
        )


def _build_initial_state(query: str, data_path: Optional[str]) -> Dict[str, Any]:
    return {
        "query": query.strip(),
        "data_path": data_path,
        "memory_context": "",
        "rag_context": "",
        "plan": [],
        "internal_tasks": [],
        "external_tasks": [],
        "coding_output": "",
        "research_output": "",
        "final_report": "",
        "error": None,
    }


@app.get("/api/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/api/memory")
def get_memory() -> Dict[str, Any]:
    if memory_store is None:
        return {"count": 0, "recent": []}

    try:
        recent = memory_store.get_recent_sessions(limit=5)
        count = memory_store.session_count()
    except Exception:
        return {"count": 0, "recent": []}

    sanitized = []
    for session in recent or []:
        sanitized.append(
            {
                "timestamp": session.get("timestamp", ""),
                "query": session.get("query", ""),
                "has_internal": bool(session.get("has_internal")),
                "has_research": bool(session.get("has_research")),
            }
        )

    return {"count": count, "recent": sanitized}


@app.get("/api/memory/list")
def list_memory(limit: int = 25, offset: int = 0, include_document: bool = False) -> Dict[str, Any]:
    if memory_store is None:
        return {"count": 0, "items": []}

    try:
        count = memory_store.session_count()
        items = memory_store.list_sessions(
            limit=limit,
            offset=offset,
            include_document=include_document,
        )
    except Exception:
        return {"count": 0, "items": []}

    sanitized = []
    for item in items:
        meta = item.get("metadata", {}) or {}
        doc = item.get("document", "") or ""
        sanitized.append(
            {
                "id": item.get("id"),
                "timestamp": meta.get("timestamp", ""),
                "query": meta.get("query", ""),
                "has_internal": bool(meta.get("has_internal")),
                "has_research": bool(meta.get("has_research")),
                "report_length": meta.get("report_length", 0),
                "excerpt": doc[:400] if include_document else "",
            }
        )

    return {"count": count, "items": sanitized}


@app.get("/api/memory/search")
def search_memory(query: str, top_k: int = 5) -> Dict[str, Any]:
    if memory_store is None:
        return {"items": []}

    if not query.strip():
        return {"items": []}

    try:
        results = memory_store.search_sessions(query_text=query, top_k=top_k)
    except Exception:
        return {"items": []}

    items = []
    for session in results:
        meta = session.get("metadata", {}) or {}
        doc = session.get("document", "") or ""
        items.append(
            {
                "id": session.get("id"),
                "timestamp": meta.get("timestamp", ""),
                "query": meta.get("query", ""),
                "has_internal": bool(meta.get("has_internal")),
                "has_research": bool(meta.get("has_research")),
                "report_length": meta.get("report_length", 0),
                "distance": session.get("distance", None),
                "excerpt": doc[:400],
            }
        )

    return {"items": items}


@app.get("/api/memory/session")
def get_memory_session(session_id: str) -> Dict[str, Any]:
    if memory_store is None:
        raise HTTPException(status_code=404, detail="Memory store unavailable.")

    if not session_id.strip():
        raise HTTPException(status_code=400, detail="session_id is required.")

    session = memory_store.get_session_by_id(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")

    meta = session.get("metadata", {}) or {}
    doc = session.get("document", "") or ""

    return {
        "id": session.get("id"),
        "timestamp": meta.get("timestamp", ""),
        "query": meta.get("query", ""),
        "has_internal": bool(meta.get("has_internal")),
        "has_research": bool(meta.get("has_research")),
        "report_length": meta.get("report_length", 0),
        "document": doc,
    }


@app.post("/api/clear-memory")
def clear_memory() -> Dict[str, Any]:
    if memory_store is None:
        return {"deleted": 0}

    try:
        deleted = memory_store.clear_all_sessions()
    except Exception:
        deleted = 0

    return {"deleted": deleted}


@app.post("/api/run")
async def run_analysis(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None),
) -> Dict[str, Any]:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required.")

    _ensure_api_key()

    data_path = None
    if file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()
        data_path = tmp.name

    graph = build_graph()
    initial_state = _build_initial_state(query=query, data_path=data_path)

    events: List[str] = []
    plan: List[Dict[str, Any]] = []
    memory_context = ""
    rag_context = ""
    coding_output = ""
    research_output = ""
    final_report = ""

    try:
        for event in graph.stream(initial_state):
            node_name = list(event.keys())[0]
            node_output = event[node_name] or {}
            events.append(node_name)

            if node_name == "memory_agent":
                memory_context = node_output.get("memory_context", "")
            elif node_name == "planner":
                plan = node_output.get("plan", [])
            elif node_name == "coding_agent":
                coding_output = node_output.get("coding_output", "")
            elif node_name == "research_agent":
                research_output = node_output.get("research_output", "")
            elif node_name == "rag_agent":
                rag_context = node_output.get("rag_context", "")
            elif node_name == "compiler":
                final_report = node_output.get("final_report", "")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "events": events,
        "plan": plan,
        "memory_context": memory_context,
        "rag_context": rag_context,
        "coding_output": coding_output,
        "research_output": research_output,
        "final_report": final_report,
    }


@app.post("/api/stream")
async def stream_analysis(
    query: str = Form(...),
    file: Optional[UploadFile] = File(None),
) -> StreamingResponse:
    if not query.strip():
        raise HTTPException(status_code=400, detail="Query is required.")

    _ensure_api_key()

    data_path = None
    if file is not None:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".csv")
        content = await file.read()
        tmp.write(content)
        tmp.flush()
        tmp.close()
        data_path = tmp.name

    graph = build_graph()
    initial_state = _build_initial_state(query=query, data_path=data_path)

    def _serialize_event(event_type: str, payload: Dict[str, Any]) -> str:
        return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"

    def _event_stream():
        events: List[str] = []
        result: Dict[str, Any] = {
            "events": events,
            "plan": [],
            "memory_context": "",
            "rag_context": "",
            "coding_output": "",
            "research_output": "",
            "final_report": "",
        }

        yield _serialize_event("status", {"message": "Pipeline started"})

        try:
            for event in graph.stream(initial_state):
                node_name = list(event.keys())[0]
                node_output = event[node_name] or {}
                events.append(node_name)

                if node_name == "memory_agent":
                    result["memory_context"] = node_output.get("memory_context", "")
                elif node_name == "planner":
                    result["plan"] = node_output.get("plan", [])
                elif node_name == "coding_agent":
                    result["coding_output"] = node_output.get("coding_output", "")
                elif node_name == "research_agent":
                    result["research_output"] = node_output.get("research_output", "")
                elif node_name == "rag_agent":
                    result["rag_context"] = node_output.get("rag_context", "")
                elif node_name == "compiler":
                    result["final_report"] = node_output.get("final_report", "")

                yield _serialize_event(
                    "node",
                    {"node": node_name, "partial": result},
                )

            yield _serialize_event("done", result)
        except Exception as exc:
            yield _serialize_event("error", {"message": str(exc)})

    return StreamingResponse(_event_stream(), media_type="text/event-stream")