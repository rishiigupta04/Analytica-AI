const API_BASE = import.meta.env.VITE_API_BASE_URL || "http://localhost:8000";

export async function runAnalysis({ query, file }) {
  const formData = new FormData();
  formData.append("query", query);
  if (file) {
    formData.append("file", file);
  }

  const response = await fetch(`${API_BASE}/api/run`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to run analysis.");
  }

  return response.json();
}

export async function runAnalysisStream({ query, file, onEvent }) {
  const formData = new FormData();
  formData.append("query", query);
  if (file) {
    formData.append("file", file);
  }

  const response = await fetch(`${API_BASE}/api/stream`, {
    method: "POST",
    body: formData,
    headers: {
      Accept: "text/event-stream",
    },
  });

  if (!response.ok || !response.body) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to stream analysis.");
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const rawEvent = buffer.slice(0, boundary).trim();
      buffer = buffer.slice(boundary + 2);

      if (rawEvent) {
        const lines = rawEvent.split("\n");
        let eventType = "message";
        let dataLine = "";

        for (const line of lines) {
          if (line.startsWith("event:")) {
            eventType = line.replace("event:", "").trim();
          } else if (line.startsWith("data:")) {
            dataLine += line.replace("data:", "").trim();
          }
        }

        if (dataLine) {
          const payload = JSON.parse(dataLine);
          onEvent?.(eventType, payload);
        }
      }

      boundary = buffer.indexOf("\n\n");
    }
  }
}

export async function fetchMemory() {
  const response = await fetch(`${API_BASE}/api/memory`);
  if (!response.ok) {
    return { count: 0, recent: [] };
  }
  return response.json();
}

export async function fetchMemoryList({ limit = 10, offset = 0, includeDocument = false }) {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
    include_document: String(includeDocument),
  });

  const response = await fetch(`${API_BASE}/api/memory/list?${params.toString()}`);
  if (!response.ok) {
    return { count: 0, items: [] };
  }
  return response.json();
}

export async function fetchMemorySession(sessionId) {
  const params = new URLSearchParams({ session_id: sessionId });
  const response = await fetch(`${API_BASE}/api/memory/session?${params.toString()}`);
  if (!response.ok) {
    const error = await response.json().catch(() => ({}));
    throw new Error(error.detail || "Failed to load session.");
  }
  return response.json();
}

export async function searchMemory({ query, topK = 5 }) {
  const params = new URLSearchParams({
    query,
    top_k: String(topK),
  });

  const response = await fetch(`${API_BASE}/api/memory/search?${params.toString()}`);
  if (!response.ok) {
    return { items: [] };
  }
  return response.json();
}

export async function clearMemory() {
  const response = await fetch(`${API_BASE}/api/clear-memory`, {
    method: "POST",
  });
  if (!response.ok) {
    return { deleted: 0 };
  }
  return response.json();
}