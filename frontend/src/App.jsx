import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { clearMemory, fetchMemory, fetchMemoryList, fetchMemorySession, runAnalysisStream, searchMemory } from "./api";

const steps = [
  { id: "memory_agent", label: "Memory agent" },
  { id: "planner", label: "Planner" },
  { id: "coding_agent", label: "Data analysis" },
  { id: "research_agent", label: "Web research" },
  { id: "rag_agent", label: "RAG context" },
  { id: "compiler", label: "Compiler" },
  { id: "save_session", label: "Save session" },
];

const tabs = [
  { id: "plan", label: "Plan" },
  { id: "memory", label: "Memory + RAG" },
  { id: "data", label: "Data analysis" },
  { id: "research", label: "Research" },
  { id: "report", label: "Final report" },
];

function classNames(...classes) {
  return classes.filter(Boolean).join(" ");
}

function formatTimestamp(value) {
  if (!value) return "";
  return value.replace("T", " ").slice(0, 16);
}

export default function App() {
  const [query, setQuery] = useState("");
  const [file, setFile] = useState(null);
  const [isRunning, setIsRunning] = useState(false);
  const [error, setError] = useState("");
  const [result, setResult] = useState(null);
  const [memory, setMemory] = useState({ count: 0, recent: [] });
  const [activeTab, setActiveTab] = useState("report");
  const [streamStatus, setStreamStatus] = useState("");
  const [memoryList, setMemoryList] = useState({ count: 0, items: [] });
  const [memoryOffset, setMemoryOffset] = useState(0);
  const [memoryQuery, setMemoryQuery] = useState("");
  const [memorySearch, setMemorySearch] = useState(null);
  const [expandedIds, setExpandedIds] = useState({});
  const [isLoadingMemory, setIsLoadingMemory] = useState(false);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [drawerLoading, setDrawerLoading] = useState(false);
  const [drawerSession, setDrawerSession] = useState(null);
  const [drawerError, setDrawerError] = useState("");

  const completed = useMemo(() => {
    const set = new Set();
    if (result?.events) {
      result.events.forEach((node) => set.add(node));
    }
    return set;
  }, [result]);

  const progressPercent = Math.round((completed.size / steps.length) * 100);

  const memoryItems = memorySearch ? memorySearch.items : memoryList.items;
  const isSearchMode = Boolean(memorySearch);

  function parseSessionDocument(documentText) {
    if (!documentText) {
      return { query: "", internal: "", research: "", report: "" };
    }

    const queryMarker = "MARKETING QUERY:";
    const internalMarker = "INTERNAL DATA FINDINGS:";
    const researchMarker = "MARKET RESEARCH FINDINGS:";
    const reportMarker = "REPORT SUMMARY:";

    const queryStart = documentText.indexOf(queryMarker);
    const internalStart = documentText.indexOf(internalMarker);
    const researchStart = documentText.indexOf(researchMarker);
    const reportStart = documentText.indexOf(reportMarker);

    const query = queryStart !== -1
      ? documentText.slice(
          queryStart + queryMarker.length,
          internalStart !== -1 ? internalStart : documentText.length
        )
      : "";

    const internal = internalStart !== -1
      ? documentText.slice(
          internalStart + internalMarker.length,
          researchStart !== -1 ? researchStart : documentText.length
        )
      : "";

    const research = researchStart !== -1
      ? documentText.slice(
          researchStart + researchMarker.length,
          reportStart !== -1 ? reportStart : documentText.length
        )
      : "";

    const report = reportStart !== -1
      ? documentText.slice(reportStart + reportMarker.length)
      : "";

    return {
      query: query.trim(),
      internal: internal.trim(),
      research: research.trim(),
      report: report.trim(),
    };
  }

  useEffect(() => {
    fetchMemory().then(setMemory).catch(() => setMemory({ count: 0, recent: [] }));
  }, []);

  useEffect(() => {
    setIsLoadingMemory(true);
    fetchMemoryList({ limit: 8, offset: 0, includeDocument: true })
      .then((data) => {
        setMemoryList(data);
        setMemoryOffset(8);
      })
      .finally(() => setIsLoadingMemory(false));
  }, []);

  async function handleRun() {
    if (!query.trim()) {
      setError("Please enter a question before running.");
      return;
    }

    setError("");
    setIsRunning(true);
    setResult(null);
    setStreamStatus("Starting pipeline...");

    try {
      await runAnalysisStream({
        query,
        file,
        onEvent: async (eventType, payload) => {
          if (eventType === "status") {
            setStreamStatus(payload.message || "Pipeline running...");
          }
          if (eventType === "node") {
            setResult(payload.partial);
            setStreamStatus(`Completed: ${payload.node.replace("_", " ")}`);
          }
          if (eventType === "done") {
            setResult(payload);
            setStreamStatus("Complete");
            const latestMemory = await fetchMemory();
            setMemory(latestMemory);
            const listData = await fetchMemoryList({ limit: 8, offset: 0, includeDocument: true });
            setMemoryList(listData);
            setMemoryOffset(8);
            setMemorySearch(null);
          }
          if (eventType === "error") {
            setError(payload.message || "Pipeline failed.");
          }
        },
      });
    } catch (err) {
      setError(err.message || "Failed to run analysis.");
    } finally {
      setIsRunning(false);
    }
  }

  async function handleClearMemory() {
    const cleared = await clearMemory();
    setMemory({ count: Math.max(0, memory.count - cleared.deleted), recent: [] });
    const listData = await fetchMemoryList({ limit: 8, offset: 0, includeDocument: true });
    setMemoryList(listData);
    setMemoryOffset(8);
    setMemorySearch(null);
  }

  async function handleSearchMemory() {
    if (!memoryQuery.trim()) {
      setMemorySearch(null);
      return;
    }

    setIsLoadingMemory(true);
    try {
      const data = await searchMemory({ query: memoryQuery, topK: 8 });
      setMemorySearch(data);
    } finally {
      setIsLoadingMemory(false);
    }
  }

  async function handleLoadMoreMemory() {
    setIsLoadingMemory(true);
    try {
      const data = await fetchMemoryList({
        limit: 8,
        offset: memoryOffset,
        includeDocument: true,
      });
      setMemoryList((prev) => ({
        count: data.count,
        items: [...prev.items, ...data.items],
      }));
      setMemoryOffset(memoryOffset + 8);
    } finally {
      setIsLoadingMemory(false);
    }
  }

  async function handleOpenSession(sessionId) {
    setDrawerOpen(true);
    setDrawerLoading(true);
    setDrawerSession(null);
    setDrawerError("");

    try {
      const session = await fetchMemorySession(sessionId);
      setDrawerSession(session);
    } catch (err) {
      setDrawerError(err.message || "Failed to load session.");
    } finally {
      setDrawerLoading(false);
    }
  }

  function handleCloseDrawer() {
    setDrawerOpen(false);
  }

  function handleDownload() {
    if (!result?.final_report) return;
    const blob = new Blob([result.final_report], { type: "text/markdown" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = "marketing_intelligence_report.md";
    link.click();
    URL.revokeObjectURL(url);
  }

  function toggleExpanded(sessionId) {
    setExpandedIds((prev) => ({
      ...prev,
      [sessionId]: !prev[sessionId],
    }));
  }

  return (
    <div className="min-h-screen bg-ink-900 text-slate-100">
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 opacity-60">
          <div className="absolute -top-20 right-0 h-64 w-64 rounded-full bg-brand-500/20 blur-[120px]" />
          <div className="absolute top-40 left-8 h-72 w-72 rounded-full bg-accent-500/10 blur-[140px]" />
        </div>
        <header className="relative mx-auto w-full max-w-7xl px-6 pt-10">
          <div className="glass rounded-3xl px-8 py-10">
            <div className="flex flex-wrap items-center justify-between gap-6">
              <div>
                <p className="text-sm uppercase tracking-[0.2em] text-brand-300">
                  Multi-Agent Marketing Intelligence
                </p>
                <h1 className="mt-3 text-3xl font-semibold text-white sm:text-4xl">
                  Marketing BI Assistant
                </h1>
                <p className="mt-3 max-w-2xl text-sm text-slate-300">
                  Premium-grade analysis, memory-aware planning, and structured intelligence
                  for modern marketing teams.
                </p>
              </div>
              <div className="badge rounded-full px-4 py-2 text-xs text-brand-300">
                Phase 2 pipeline enabled
              </div>
            </div>
          </div>
        </header>
      </div>

      <main className="mx-auto w-full max-w-7xl px-6 pb-16 pt-10">
        <div className="grid gap-6 lg:grid-cols-[320px,1fr]">
          <section className="space-y-6">
            <div className="card p-5 shadow-soft">
              <h2 className="text-base font-semibold text-white">Your marketing data</h2>
              <p className="mt-2 text-sm text-slate-400">
                Upload a CSV to unlock internal data insights. Leave empty for research-only.
              </p>
              <label className="mt-4 flex cursor-pointer flex-col gap-2 rounded-xl border border-dashed border-slate-700 bg-ink-800/60 px-4 py-5 text-sm text-slate-300">
                <span className="text-xs uppercase tracking-wide text-slate-500">
                  Upload CSV
                </span>
                <input
                  type="file"
                  accept=".csv"
                  className="hidden"
                  onChange={(event) => setFile(event.target.files?.[0] || null)}
                />
                <span>{file ? file.name : "Click to choose a file"}</span>
              </label>
              {file && (
                <div className="mt-3 text-xs text-slate-400">
                  Size: {(file.size / 1024).toFixed(1)} KB
                </div>
              )}
            </div>

            <div className="card p-5">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold text-white">Memory store</h2>
                <button
                  onClick={handleClearMemory}
                  className="text-xs text-slate-400 transition hover:text-slate-200"
                  type="button"
                >
                  Clear
                </button>
              </div>
              <p className="mt-2 text-sm text-slate-400">
                Sessions stored: <span className="text-white">{memory.count}</span>
              </p>
              <div className="mt-4 space-y-3">
                {memory.recent.length === 0 && (
                  <div className="rounded-lg border border-slate-800 bg-ink-800/60 p-3 text-xs text-slate-400">
                    No sessions yet. Run your first query to build memory.
                  </div>
                )}
                {memory.recent.map((session, index) => (
                  <div key={index} className="rounded-lg border border-slate-800 bg-ink-800/60 p-3">
                    <div className="text-xs text-slate-500">
                      {formatTimestamp(session.timestamp)}
                    </div>
                    <div className="mt-2 text-sm text-slate-200">
                      {session.query || "Unknown query"}
                    </div>
                    <div className="mt-2 text-[11px] text-slate-500">
                      {session.has_internal ? "Internal" : ""}
                      {session.has_internal && session.has_research ? " + " : ""}
                      {session.has_research ? "Research" : ""}
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <div className="card p-5">
              <h2 className="text-base font-semibold text-white">Memory browser</h2>
              <p className="mt-2 text-sm text-slate-400">
                Search past sessions or browse the full history.
              </p>
              <div className="mt-4 flex gap-2">
                <input
                  value={memoryQuery}
                  onChange={(event) => setMemoryQuery(event.target.value)}
                  placeholder="Search memory"
                  className="w-full rounded-lg border border-slate-700 bg-ink-800/70 px-3 py-2 text-xs text-slate-100 placeholder:text-slate-500 focus:border-brand-400 focus:outline-none"
                />
                <button
                  onClick={handleSearchMemory}
                  className="rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 transition hover:border-slate-500"
                  type="button"
                >
                  Search
                </button>
              </div>
              {memorySearch && (
                <button
                  onClick={() => {
                    setMemorySearch(null);
                    setMemoryQuery("");
                  }}
                  className="mt-2 text-xs text-slate-400 transition hover:text-slate-200"
                  type="button"
                >
                  Reset search
                </button>
              )}

              <div className="mt-4 space-y-3">
                {isLoadingMemory && (
                  <div className="rounded-lg border border-slate-800 bg-ink-800/60 p-3 text-xs text-slate-400">
                    Loading memory entries...
                  </div>
                )}
                {!isLoadingMemory && isSearchMode && memoryItems.length === 0 && (
                  <div className="rounded-lg border border-slate-800 bg-ink-800/60 p-3 text-xs text-slate-400">
                    No matches for "{memoryQuery}".
                  </div>
                )}
                {!isLoadingMemory && !isSearchMode && memoryItems.length === 0 && (
                  <div className="rounded-lg border border-slate-800 bg-ink-800/60 p-3 text-xs text-slate-400">
                    No sessions found yet.
                  </div>
                )}
                {memoryItems.map((session) => (
                  <div key={session.id} className="rounded-lg border border-slate-800 bg-ink-800/60 p-3">
                    <div className="flex items-center justify-between text-xs text-slate-500">
                      <span>{formatTimestamp(session.timestamp)}</span>
                      {session.distance !== undefined && session.distance !== null && (
                        <span>Similarity: {Number(session.distance).toFixed(3)}</span>
                      )}
                    </div>
                    <div className="mt-2 text-sm text-slate-200">
                      {session.query || "Unknown query"}
                    </div>
                    <div className="mt-2 text-[11px] text-slate-500">
                      {session.has_internal ? "Internal" : ""}
                      {session.has_internal && session.has_research ? " + " : ""}
                      {session.has_research ? "Research" : ""}
                      {session.report_length ? ` · ${session.report_length} chars` : ""}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      {session.excerpt && (
                        <button
                          onClick={() => toggleExpanded(session.id)}
                          className="text-xs text-brand-300 transition hover:text-brand-200"
                          type="button"
                        >
                          {expandedIds[session.id] ? "Hide excerpt" : "Show excerpt"}
                        </button>
                      )}
                      <button
                        onClick={() => handleOpenSession(session.id)}
                        className="text-xs text-slate-300 transition hover:text-white"
                        type="button"
                      >
                        View full session
                      </button>
                    </div>
                    {expandedIds[session.id] && session.excerpt && (
                      <div className="mt-2 text-xs text-slate-400">
                        {session.excerpt}
                      </div>
                    )}
                  </div>
                ))}
              </div>

              {!memorySearch && memoryList.items.length < memoryList.count && (
                <button
                  onClick={handleLoadMoreMemory}
                  className="mt-4 w-full rounded-lg border border-slate-700 px-3 py-2 text-xs text-slate-300 transition hover:border-slate-500"
                  type="button"
                >
                  Load more
                </button>
              )}
            </div>

            <div className="card p-5">
              <h2 className="text-base font-semibold text-white">Pipeline</h2>
              <p className="mt-2 text-sm text-slate-400">
                Live status for the seven-node intelligence workflow.
              </p>
              <div className="mt-4 space-y-2">
                {steps.map((step) => (
                  <div key={step.id} className="flex items-center justify-between text-sm">
                    <span className="text-slate-300">{step.label}</span>
                    <span
                      className={classNames(
                        "rounded-full px-2 py-1 text-[11px]",
                        completed.has(step.id)
                          ? "bg-emerald-500/20 text-emerald-200"
                          : "bg-slate-700/50 text-slate-400"
                      )}
                    >
                      {completed.has(step.id) ? "Complete" : "Pending"}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="card p-5">
              <h2 className="text-base font-semibold text-white">Example questions</h2>
              <ul className="mt-3 space-y-2 text-sm text-slate-400">
                <li>Which channel has the lowest ROI this quarter?</li>
                <li>Break down revenue by campaign type.</li>
                <li>How does our email performance compare to the industry?</li>
              </ul>
            </div>
          </section>

          <section className="space-y-6">
            <div className="card p-6">
              <div className="flex flex-wrap items-center justify-between gap-4">
                <div>
                  <h2 className="text-xl font-semibold text-white">Ask your question</h2>
                  <p className="mt-2 text-sm text-slate-400">
                    Combine internal data with market research for best results.
                  </p>
                </div>
                <button
                  onClick={handleRun}
                  className={classNames(
                    "rounded-xl px-5 py-3 text-sm font-semibold transition",
                    isRunning
                      ? "bg-slate-700 text-slate-300"
                      : "bg-brand-500 text-white hover:bg-brand-400"
                  )}
                  disabled={isRunning}
                  type="button"
                >
                  {isRunning ? "Running..." : "Run analysis"}
                </button>
              </div>
              <textarea
                value={query}
                onChange={(event) => setQuery(event.target.value)}
                placeholder="Example: Which marketing channel is underperforming and how can we improve it?"
                rows={5}
                className="mt-5 w-full rounded-xl border border-slate-700 bg-ink-800/70 px-4 py-3 text-sm text-slate-100 placeholder:text-slate-500 focus:border-brand-400 focus:outline-none"
              />
              {error && (
                <div className="mt-4 rounded-xl border border-rose-500/40 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
                  {error}
                </div>
              )}
            </div>

            <div className="card p-6">
              <div className="flex flex-wrap items-center justify-between gap-3">
                <div>
                  <h2 className="text-xl font-semibold text-white">Results</h2>
                  <p className="mt-1 text-sm text-slate-400">
                    Explore the plan, memory context, and final report in one place.
                  </p>
                  {streamStatus && (
                    <p className="mt-2 text-xs text-slate-500">Status: {streamStatus}</p>
                  )}
                  <div className="mt-3">
                    <div className="flex items-center justify-between text-[11px] text-slate-500">
                      <span>Progress</span>
                      <span>{progressPercent}%</span>
                    </div>
                    <div className="mt-2 h-2 w-full rounded-full bg-ink-800/80">
                      <div
                        className="h-2 rounded-full bg-brand-500 transition-all"
                        style={{ width: `${progressPercent}%` }}
                      />
                    </div>
                  </div>
                </div>
                <button
                  onClick={handleDownload}
                  className="rounded-xl border border-slate-700 px-4 py-2 text-xs text-slate-300 transition hover:border-slate-500"
                  type="button"
                >
                  Download report
                </button>
              </div>

              <div className="mt-6 flex flex-wrap gap-2">
                {tabs.map((tab) => (
                  <button
                    key={tab.id}
                    onClick={() => setActiveTab(tab.id)}
                    className={classNames(
                      "rounded-full px-4 py-2 text-xs font-medium transition",
                      activeTab === tab.id
                        ? "bg-brand-500 text-white"
                        : "bg-ink-800/60 text-slate-400 hover:text-slate-200"
                    )}
                    type="button"
                  >
                    {tab.label}
                  </button>
                ))}
              </div>

              <div className="mt-6">
                {!result && (
                  <div className="rounded-xl border border-slate-800 bg-ink-800/60 p-6 text-sm text-slate-400">
                    Run a query to generate insights and a full report.
                  </div>
                )}

                {result && activeTab === "plan" && (
                  <div className="space-y-3">
                    {(result.plan || []).length === 0 && (
                      <div className="rounded-xl border border-slate-800 bg-ink-800/60 p-4 text-sm text-slate-400">
                        No plan steps returned.
                      </div>
                    )}
                    {(result.plan || []).map((task) => (
                      <div key={task.id} className="rounded-xl border border-slate-800 bg-ink-800/60 p-4">
                        <div className="text-sm font-semibold text-white">
                          Task {task.id}
                        </div>
                        <div className="mt-2 text-sm text-slate-300">{task.task}</div>
                        <div className="mt-2 text-xs text-slate-500">Source: {task.source}</div>
                      </div>
                    ))}
                  </div>
                )}

                {result && activeTab === "memory" && (
                  <div className="space-y-4">
                    <div className="rounded-xl border border-slate-800 bg-ink-800/60 p-4">
                      <div className="text-sm font-semibold text-white">Memory context</div>
                      <div className="prose prose-invert mt-3 max-w-none text-sm">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {result.memory_context || "No memory context available."}
                        </ReactMarkdown>
                      </div>
                    </div>
                    <div className="rounded-xl border border-slate-800 bg-ink-800/60 p-4">
                      <div className="text-sm font-semibold text-white">RAG context</div>
                      <div className="prose prose-invert mt-3 max-w-none text-sm">
                        <ReactMarkdown remarkPlugins={[remarkGfm]}>
                          {result.rag_context || "No historical comparison data yet."}
                        </ReactMarkdown>
                      </div>
                    </div>
                  </div>
                )}

                {result && activeTab === "data" && (
                  <div className="rounded-xl border border-slate-800 bg-ink-800/60 p-4">
                    <div className="prose prose-invert max-w-none text-sm">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {result.coding_output || "No internal data analysis output."}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}

                {result && activeTab === "research" && (
                  <div className="rounded-xl border border-slate-800 bg-ink-800/60 p-4">
                    <div className="prose prose-invert max-w-none text-sm">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {result.research_output || "No external research output."}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}

                {result && activeTab === "report" && (
                  <div className="rounded-xl border border-slate-800 bg-ink-800/60 p-4">
                    <div className="prose prose-invert max-w-none text-sm">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {result.final_report || "Report is empty."}
                      </ReactMarkdown>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </section>
        </div>
      </main>

      {drawerOpen && (
        <div className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-black/60"
            role="button"
            tabIndex={-1}
            onClick={handleCloseDrawer}
            onKeyDown={(event) => {
              if (event.key === "Escape") handleCloseDrawer();
            }}
          />
          <aside className="absolute right-0 top-0 h-full w-full max-w-xl bg-ink-900 shadow-2xl">
            <div className="flex h-full flex-col border-l border-slate-800">
              <div className="flex items-center justify-between border-b border-slate-800 px-6 py-4">
                <div>
                  <h3 className="text-lg font-semibold text-white">Memory session</h3>
                  <p className="text-xs text-slate-400">Full session context</p>
                </div>
                <button
                  onClick={handleCloseDrawer}
                  className="rounded-full border border-slate-700 px-3 py-1 text-xs text-slate-300"
                  type="button"
                >
                  Close
                </button>
              </div>
              <div className="flex-1 overflow-y-auto px-6 py-5">
                {drawerLoading && (
                  <div className="rounded-lg border border-slate-800 bg-ink-800/60 p-4 text-sm text-slate-400">
                    Loading session...
                  </div>
                )}
                {drawerError && (
                  <div className="rounded-lg border border-rose-500/40 bg-rose-500/10 p-4 text-sm text-rose-200">
                    {drawerError}
                  </div>
                )}
                {drawerSession && (
                  <div className="space-y-4">
                    <div className="rounded-lg border border-slate-800 bg-ink-800/60 p-4 text-sm text-slate-300">
                      <div className="text-xs text-slate-500">{formatTimestamp(drawerSession.timestamp)}</div>
                      <div className="mt-2 text-base text-white">{drawerSession.query}</div>
                      <div className="mt-2 text-xs text-slate-500">
                        {drawerSession.has_internal ? "Internal" : ""}
                        {drawerSession.has_internal && drawerSession.has_research ? " + " : ""}
                        {drawerSession.has_research ? "Research" : ""}
                        {drawerSession.report_length ? ` · ${drawerSession.report_length} chars` : ""}
                      </div>
                    </div>
                    {(() => {
                      const sections = parseSessionDocument(drawerSession.document || "");
                      const sectionList = [
                        { label: "Query", value: sections.query },
                        { label: "Internal", value: sections.internal },
                        { label: "Research", value: sections.research },
                        { label: "Report", value: sections.report },
                      ];

                      return (
                        <div className="space-y-3">
                          {sectionList.map((section) => (
                            <div
                              key={section.label}
                              className="rounded-lg border border-slate-800 bg-ink-800/60 p-4"
                            >
                              <div className="text-sm font-semibold text-white">{section.label}</div>
                              <div className="prose prose-invert mt-3 max-w-none text-sm">
                                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                                  {section.value || `No ${section.label.toLowerCase()} content.`}
                                </ReactMarkdown>
                              </div>
                            </div>
                          ))}
                        </div>
                      );
                    })()}
                  </div>
                )}
              </div>
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}