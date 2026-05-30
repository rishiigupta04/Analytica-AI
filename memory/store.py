# =============================================================
# memory/store.py — ChromaDB Persistent Memory Store
# =============================================================
# ROLE: The long-term memory database for the entire system.
#
# WHAT IT DOES:
#   - Persists every completed session (query + findings + report)
#     into a local ChromaDB vector database at ./memory_db/
#   - Converts text into semantic vector embeddings using
#     the free HuggingFace model "all-MiniLM-L6-v2"
#   - Enables semantic search: "find sessions similar to this query"
#     even if the wording is completely different
#
# KEY CONCEPT — Vector Embeddings:
#   Every piece of text gets turned into a list of ~384 numbers
#   (a "vector") that captures its MEANING. Two texts about the
#   same topic will have similar vectors even if they use different
#   words. ChromaDB stores these vectors and lets us search by
#   similarity — this is what makes RAG intelligent.
#
# IMPORTANT — First Run:
#   On first use, sentence-transformers will download the
#   all-MiniLM-L6-v2 model (~90MB). This happens once and
#   is then cached locally. Subsequent uses are instant.
#
# DATA STORED PER SESSION:
#   - Document: combined text of query + findings + report (searchable)
#   - Metadata:  query text, timestamp, flags, report length
#   - ID:        unique string per session
# =============================================================

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional

import chromadb
from sentence_transformers import SentenceTransformer

# ── Paths ──────────────────────────────────────────────────────
# memory/store.py → parent = memory/ → parent = project root
DB_PATH = Path(__file__).resolve().parent.parent / "memory_db"
DB_PATH.mkdir(parents=True, exist_ok=True)

# ── Model Init (module-level singleton) ────────────────────────
# Loading the model once at module import avoids reloading on
# every agent call. The first load downloads ~90MB from HuggingFace.
print("🧠 [MEMORY] Loading sentence-transformer embedding model...")
_embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
print("   ✅ Embedding model ready.")

# ── ChromaDB Init ──────────────────────────────────────────────
_client = chromadb.PersistentClient(path=str(DB_PATH))

# Get or create the sessions collection.
# metadata={"hnsw:space": "cosine"} → uses cosine similarity for text
# (better than euclidean distance for comparing sentence meanings)
_collection = _client.get_or_create_collection(
    name="marketing_sessions",
    metadata={"hnsw:space": "cosine"},
)


# ──────────────────────────────────────────────────────────────
# PUBLIC FUNCTIONS
# ──────────────────────────────────────────────────────────────

def save_session(
    query: str,
    coding_output: str,
    research_output: str,
    final_report: str,
) -> str:
    """
    Saves a completed pipeline session into ChromaDB.

    The saved document combines key text so semantic search can
    find this session when similar questions arise in the future.

    Args:
        query:           The marketer's original question
        coding_output:   Internal data findings from coding agent
        research_output: Web research findings from research agent
        final_report:    The compiled final report

    Returns:
        session_id: The unique ID assigned to this session
    """
    # Create a unique session ID: timestamp + hash of query
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    query_hash = hashlib.md5(query.encode()).hexdigest()[:8]
    session_id = f"session_{ts}_{query_hash}"

    # Build the document text that will be embedded and searched.
    # We truncate each section to stay within reasonable size limits.
    document_text = (
        f"MARKETING QUERY: {query}\n\n"
        f"INTERNAL DATA FINDINGS:\n{coding_output[:800]}\n\n"
        f"MARKET RESEARCH FINDINGS:\n{research_output[:800]}\n\n"
        f"REPORT SUMMARY:\n{final_report[:1200]}"
    )

    # Convert to embedding vector
    embedding = _embedding_model.encode([document_text]).tolist()

    # ChromaDB metadata: must be flat (no nested dicts)
    # Values must be str, int, float, or bool
    metadata = {
        "query":         query[:500],
        "timestamp":     datetime.now().isoformat(),
        "has_internal":  int("No internal data" not in coding_output),
        "has_research":  int("No external research" not in research_output),
        "report_length": len(final_report),
    }

    _collection.add(
        documents=[document_text],
        embeddings=embedding,
        metadatas=[metadata],
        ids=[session_id],
    )

    print(f"   💾 [MEMORY STORE] Session saved: {session_id}")
    return session_id


def search_sessions(query_text: str, top_k: int = 3) -> list[dict]:
    """
    Semantic search for past sessions similar to the given text.

    Unlike keyword search, this finds sessions that are
    CONCEPTUALLY similar even if they use different words.
    Example: "worst channel" finds sessions about "lowest performance".

    Args:
        query_text: Text to search for similar sessions
        top_k:      Max number of results to return

    Returns:
        List of dicts, each with:
            - document: full stored text
            - metadata: session metadata (query, timestamp, etc.)
            - distance: similarity score (lower = more similar for cosine)
    """
    total_docs = _collection.count()
    if total_docs == 0:
        return []  # Empty store — no past sessions yet

    # Clamp k to available documents (ChromaDB errors if k > n docs)
    k = min(top_k, total_docs)

    # Embed the search query
    query_embedding = _embedding_model.encode([query_text]).tolist()

    results = _collection.query(
        query_embeddings=query_embedding,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    # Reformat into a clean list of dicts
    sessions = []
    for i in range(len(results["ids"][0])):
        sessions.append({
            "id":       results["ids"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return sessions


def get_recent_sessions(limit: int = 10) -> list[dict]:
    """
    Returns the most recent sessions for sidebar display in the UI.

    Note: ChromaDB doesn't sort by timestamp natively, so we sort
    the results in Python after fetching them.

    Returns:
        List of metadata dicts, sorted newest-first
    """
    total = _collection.count()
    if total == 0:
        return []

    result = _collection.get(
        limit=min(limit, total),
        include=["metadatas"],
    )

    # Sort by timestamp string (ISO format sorts correctly lexicographically)
    metadatas = result.get("metadatas", [])
    return sorted(metadatas, key=lambda x: x.get("timestamp", ""), reverse=True)


def clear_all_sessions() -> int:
    """
    Deletes all sessions from the memory store.
    Used by the "Clear Memory" button in the UI.

    Returns:
        Number of sessions deleted
    """
    all_ids = _collection.get()["ids"]
    count = len(all_ids)

    if all_ids:
        _collection.delete(ids=all_ids)
        print(f"   🗑️  [MEMORY STORE] Cleared {count} session(s).")

    return count


def session_count() -> int:
    """Returns the total number of stored sessions."""
    return _collection.count()