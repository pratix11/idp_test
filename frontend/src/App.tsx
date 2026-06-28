import { useCallback, useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import {
  Bot,
  Building2,
  GitCompareArrows,
  MessageSquare,
  Search as SearchIcon,
  Send,
  Sparkles,
  Zap,
  ChevronDown,
  FileText,
  Loader2,
  AlertTriangle,
  ScrollText,
  ShieldAlert,
} from "lucide-react";
import { Toaster } from "@/components/ui/sonner";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Badge } from "@/components/ui/badge";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  ask,
  summarize as apiSummarize,
  compare as apiCompare,
  search as apiSearch,
  runAgent,
  checkHealth,
  handleApiError,
  ApiError,
  type Citation,
  type SearchResult,
  type Role,
} from "@/lib/api";
import { cn } from "@/lib/utils";

type Tab = "chat" | "summarize" | "compare" | "search" | "agent";
type HealthState = "checking" | "online" | "offline";

const ROLE_PERMISSIONS: Record<Role, Tab[]> = {
  admin: ["chat", "summarize", "compare", "search", "agent"],
  analyst: ["chat", "summarize", "compare", "search", "agent"],
  viewer: ["search"],
  auditor: ["search"],
};

const ROLE_BADGE: Record<Role, string> = {
  admin: "bg-red-500/20 text-red-300 border-red-500/30",
  analyst: "bg-primary/20 text-primary border-primary/30",
  viewer: "bg-muted text-muted-foreground border-border",
  auditor: "bg-yellow-500/20 text-yellow-300 border-yellow-500/30",
};

// =====================================================
// Root
// =====================================================
export default function PropIntelApp() {
  const [tab, setTab] = useState<Tab>("chat");
  const [role, setRole] = useState<Role>("analyst");
  const [health, setHealth] = useState<HealthState>("checking");

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      const ok = await checkHealth();
      if (!cancelled) setHealth(ok ? "online" : "offline");
    };
    run();
    const id = setInterval(run, 15000);
    return () => {
      cancelled = true;
      clearInterval(id);
    };
  }, []);

  const allowed = ROLE_PERMISSIONS[role].includes(tab);

  return (
    <div className="flex h-screen overflow-hidden bg-background text-foreground">
      <Toaster theme="dark" position="top-right" />
      <Sidebar tab={tab} setTab={setTab} role={role} setRole={setRole} health={health} />
      <main className="flex-1 min-w-0 overflow-hidden flex flex-col">
        {health === "offline" && (
          <div className="bg-yellow-500/10 border-b border-yellow-500/20 px-6 py-2.5 text-sm text-yellow-300 flex items-center gap-2">
            <AlertTriangle className="h-4 w-4" />
            Backend offline — start{" "}
            <code className="px-1.5 py-0.5 bg-yellow-500/10 rounded text-xs font-mono">
              uv run uvicorn property_intel.api.app:app --reload
            </code>{" "}
            to enable AI features.
          </div>
        )}
        {!allowed ? (
          <PermissionDenied role={role} tab={tab} />
        ) : (
          <>
            {tab === "chat" && <ChatPanel role={role} />}
            {tab === "summarize" && <SummarizePanel role={role} />}
            {tab === "compare" && <ComparePanel role={role} />}
            {tab === "search" && <SearchPanel role={role} />}
            {tab === "agent" && <AgentPanel role={role} />}
          </>
        )}
      </main>
    </div>
  );
}

function PermissionDenied({ role, tab }: { role: Role; tab: Tab }) {
  return (
    <div className="flex-1 flex items-center justify-center p-6">
      <div className="max-w-md text-center">
        <div className="mx-auto h-12 w-12 rounded-full bg-yellow-500/10 border border-yellow-500/20 flex items-center justify-center mb-4">
          <ShieldAlert className="h-6 w-6 text-yellow-400" />
        </div>
        <h2 className="text-lg font-semibold">Permission required</h2>
        <p className="mt-2 text-sm text-muted-foreground">
          Your current role (<strong className="text-foreground">{role}</strong>) doesn't have
          permission to use <strong className="text-foreground">{tab}</strong>. Switch to{" "}
          <strong className="text-foreground">analyst</strong> or{" "}
          <strong className="text-foreground">admin</strong> in the sidebar.
        </p>
      </div>
    </div>
  );
}

// =====================================================
// Sidebar
// =====================================================
function Sidebar({
  tab,
  setTab,
  role,
  setRole,
  health,
}: {
  tab: Tab;
  setTab: (t: Tab) => void;
  role: Role;
  setRole: (r: Role) => void;
  health: HealthState;
}) {
  const items: {
    id: Tab;
    label: string;
    icon: React.ComponentType<{ className?: string }>;
    badge?: string;
    accent?: "agent";
  }[] = [
    { id: "chat", label: "Chat", icon: MessageSquare },
    { id: "summarize", label: "Summarize", icon: ScrollText },
    { id: "compare", label: "Compare", icon: GitCompareArrows },
    { id: "search", label: "Search", icon: SearchIcon },
    { id: "agent", label: "Agent", icon: Zap, badge: "Phase 5", accent: "agent" },
  ];

  return (
    <aside className="w-[240px] shrink-0 border-r border-sidebar-border bg-sidebar flex flex-col">
      {/* Logo */}
      <div className="px-5 pt-5 pb-5">
        <div className="flex items-center gap-3">
          <div className="h-9 w-9 rounded-lg bg-primary/20 border border-primary/40 flex items-center justify-center">
            <Building2 className="h-5 w-5 text-primary" />
          </div>
          <div>
            <div className="text-[15px] font-semibold tracking-tight text-sidebar-foreground">
              PropIntel
            </div>
            <div className="text-[11px] text-muted-foreground">AI Copilot</div>
          </div>
        </div>
      </div>

      {/* Nav */}
      <nav className="px-3 space-y-0.5 flex-1">
        {items.map((it) => {
          const Icon = it.icon;
          const active = tab === it.id;
          const isAgent = it.accent === "agent";
          return (
            <button
              key={it.id}
              onClick={() => setTab(it.id)}
              className={cn(
                "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
                active
                  ? isAgent
                    ? "bg-violet-500/15 text-violet-300 border border-violet-500/25"
                    : "bg-primary/15 text-primary border border-primary/25"
                  : "text-sidebar-foreground/70 hover:bg-sidebar-accent hover:text-sidebar-foreground border border-transparent",
              )}
            >
              <Icon
                className={cn(
                  "h-4 w-4 shrink-0",
                  active && (isAgent ? "text-violet-400" : "text-primary"),
                )}
              />
              <span className="font-medium flex-1 text-left">{it.label}</span>
              {it.badge && (
                <span
                  className={cn(
                    "text-[9px] font-semibold px-1.5 py-0.5 rounded tracking-wide",
                    isAgent
                      ? "bg-violet-500/20 text-violet-300"
                      : "bg-muted text-muted-foreground",
                  )}
                >
                  {it.badge}
                </span>
              )}
            </button>
          );
        })}
      </nav>

      {/* Role + status */}
      <div className="border-t border-sidebar-border p-4 space-y-3">
        <div>
          <div className="text-[10px] font-semibold tracking-[0.14em] text-muted-foreground mb-1.5">
            ROLE
          </div>
          <Select value={role} onValueChange={(v) => setRole(v as Role)}>
            <SelectTrigger className="h-9 text-sm bg-sidebar border-sidebar-border">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {(["admin", "analyst", "viewer", "auditor"] as Role[]).map((r) => (
                <SelectItem key={r} value={r}>
                  <div className="flex items-center gap-2">
                    <span
                      className={cn(
                        "h-1.5 w-1.5 rounded-full",
                        r === "admin" && "bg-red-400",
                        r === "analyst" && "bg-primary",
                        r === "viewer" && "bg-muted-foreground",
                        r === "auditor" && "bg-yellow-400",
                      )}
                    />
                    <span className="capitalize">{r}</span>
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <div className="mt-2">
            <span
              className={cn(
                "inline-flex items-center text-[10px] font-semibold uppercase tracking-wide px-2 py-0.5 rounded border",
                ROLE_BADGE[role],
              )}
            >
              {role}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          <span
            className={cn(
              "h-2 w-2 rounded-full",
              health === "online" && "bg-emerald-400 shadow-[0_0_6px] shadow-emerald-400/60",
              health === "offline" && "bg-red-500",
              health === "checking" && "bg-muted-foreground animate-pulse",
            )}
          />
          <span>
            {health === "online"
              ? "Connected"
              : health === "offline"
                ? "Offline"
                : "Checking…"}
          </span>
        </div>
      </div>
    </aside>
  );
}

// =====================================================
// Shared: citations + answer card
// =====================================================
function CitationsBlock({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(true);
  if (!citations?.length) return null;
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-4">
      <CollapsibleTrigger asChild>
        <button className="inline-flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors">
          <FileText className="h-3.5 w-3.5" />
          {citations.length} citation{citations.length !== 1 ? "s" : ""}
          <ChevronDown
            className={cn("h-3.5 w-3.5 transition-transform", open && "rotate-180")}
          />
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-2 space-y-2">
        {citations.map((c) => (
          <div
            key={`${c.index}-${c.chunk_id}`}
            className="rounded-md border border-border bg-card p-3"
          >
            <div className="flex items-start gap-2">
              <span className="shrink-0 inline-flex items-center justify-center min-w-[1.5rem] h-6 px-1.5 rounded bg-primary/15 text-primary text-xs font-semibold">
                {c.index}
              </span>
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium text-foreground truncate">
                  {c.document_title || "Untitled"}
                </div>
                {c.section_title && (
                  <div className="text-xs text-muted-foreground mt-0.5">{c.section_title}</div>
                )}
                <p className="text-xs text-foreground/80 mt-1.5 line-clamp-3 leading-relaxed">
                  {c.content_snippet}
                </p>
              </div>
            </div>
          </div>
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}

function AnswerCard({ answer, citations }: { answer: string; citations: Citation[] }) {
  return (
    <div className="rounded-lg border border-border bg-card p-5">
      <div className="flex items-center gap-2 mb-3">
        <div className="h-6 w-6 rounded-full bg-primary/15 flex items-center justify-center">
          <Bot className="h-3.5 w-3.5 text-primary" />
        </div>
        <span className="text-xs font-semibold text-foreground/60 tracking-wide uppercase">
          Answer
        </span>
      </div>
      <div className="text-sm text-foreground/90 leading-relaxed [&_strong]:text-foreground [&_h1]:text-base [&_h2]:text-sm [&_h3]:text-sm [&_ul]:list-disc [&_ul]:pl-4 [&_ol]:list-decimal [&_ol]:pl-4 [&_li]:mt-1 [&_p]:mb-2">
        <ReactMarkdown>{answer}</ReactMarkdown>
      </div>
      <CitationsBlock citations={citations} />
    </div>
  );
}

function ErrorCard({ message }: { message: string }) {
  return (
    <div className="rounded-lg border border-yellow-500/20 bg-yellow-500/10 p-4 flex items-start gap-3">
      <AlertTriangle className="h-4 w-4 text-yellow-400 mt-0.5 shrink-0" />
      <p className="text-sm text-yellow-200">{message}</p>
    </div>
  );
}

function PageHeader({
  title,
  subtitle,
  accent,
}: {
  title: string;
  subtitle?: string;
  accent?: "agent";
}) {
  return (
    <div className="border-b border-border bg-card/50 px-8 py-5">
      <h1
        className={cn(
          "text-xl font-semibold tracking-tight",
          accent === "agent" ? "text-violet-300" : "text-foreground",
        )}
      >
        {title}
      </h1>
      {subtitle && <p className="text-sm text-muted-foreground mt-1">{subtitle}</p>}
    </div>
  );
}

// =====================================================
// Chat Panel
// =====================================================
const SAMPLE_QUESTIONS = [
  "What are the project registration requirements under MahaRERA?",
  "Can a promoter accept advance payments before signing an agreement?",
  "What disclosures must be filed quarterly under MahaRERA?",
];

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
}

function ChatPanel({ role }: { role: Role }) {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const inputRef = useRef<HTMLTextAreaElement | null>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, busy]);

  const send = useCallback(
    async (text: string) => {
      const q = text.trim();
      if (!q || busy) return;
      setError(null);
      setBusy(true);
      setMessages((p) => [...p, { id: crypto.randomUUID(), role: "user", content: q }]);
      setInput("");
      try {
        const res = await ask(q, role);
        setMessages((p) => [
          ...p,
          {
            id: crypto.randomUUID(),
            role: "assistant",
            content: res.answer,
            citations: res.citations,
          },
        ]);
      } catch (e) {
        setError(handleApiError(e));
      } finally {
        setBusy(false);
        inputRef.current?.focus();
      }
    },
    [busy, role],
  );

  const onKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send(input);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Chat" subtitle="Ask a question about Indian property regulations." />

      <div className="border-b border-border bg-card/30 px-8 py-4">
        <div className="max-w-3xl mx-auto">
          <div className="flex items-end gap-2">
            <Textarea
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={onKeyDown}
              placeholder="Ask a question about MahaRERA, RERA Act, circulars…"
              className="min-h-[52px] resize-none"
            />
            <Button
              onClick={() => send(input)}
              disabled={busy || !input.trim()}
              className="h-[52px] px-5"
            >
              {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
              <span className="ml-2 font-medium">Ask</span>
            </Button>
          </div>
          {messages.length === 0 && (
            <div className="mt-3 flex flex-wrap gap-2">
              {SAMPLE_QUESTIONS.map((q) => (
                <button
                  key={q}
                  onClick={() => setInput(q)}
                  className="text-xs px-3 py-1.5 rounded-full border border-border bg-card hover:border-primary/40 hover:bg-primary/5 text-foreground/70 transition-colors"
                >
                  {q}
                </button>
              ))}
            </div>
          )}
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-4">
          {messages.length === 0 ? (
            <EmptyState
              icon={Sparkles}
              title="Ask your property law copilot"
              description="Get cited answers grounded in MahaRERA, the RERA Act, and official circulars."
            />
          ) : (
            messages.map((m) =>
              m.role === "user" ? (
                <div key={m.id} className="flex justify-end">
                  <div className="max-w-[80%] rounded-2xl rounded-tr-sm px-4 py-2.5 bg-primary text-primary-foreground">
                    <p className="text-sm whitespace-pre-wrap leading-relaxed">{m.content}</p>
                  </div>
                </div>
              ) : (
                <AnswerCard key={m.id} answer={m.content} citations={m.citations || []} />
              ),
            )
          )}
          {busy && (
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Thinking…
            </div>
          )}
          {error && <ErrorCard message={error} />}
        </div>
      </div>
    </div>
  );
}

function EmptyState({
  icon: Icon,
  title,
  description,
}: {
  icon: React.ComponentType<{ className?: string }>;
  title: string;
  description: string;
}) {
  return (
    <div className="flex flex-col items-center text-center py-14">
      <div className="h-12 w-12 rounded-xl bg-primary/10 border border-primary/20 flex items-center justify-center mb-4">
        <Icon className="h-5 w-5 text-primary" />
      </div>
      <h3 className="text-base font-semibold">{title}</h3>
      <p className="mt-1.5 text-sm text-muted-foreground max-w-md">{description}</p>
    </div>
  );
}

// =====================================================
// Summarize
// =====================================================
function SummarizePanel({ role }: { role: Role }) {
  const [query, setQuery] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ answer: string; citations: Citation[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!query.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await apiSummarize(query.trim(), role);
      setResult(r);
    } catch (e) {
      setError(handleApiError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <PageHeader
        title="Summarize"
        subtitle="Generate a concise summary of any property regulation topic."
      />
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-5">
          <div className="rounded-lg border border-border bg-card p-5">
            <label className="block text-sm font-medium mb-2">Topic or keywords to summarize</label>
            <div className="flex gap-2">
              <Input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && run()}
                placeholder="e.g. MahaRERA registration process"
              />
              <Button onClick={run} disabled={busy || !query.trim()}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : "Summarize"}
              </Button>
            </div>
          </div>
          {error && <ErrorCard message={error} />}
          {result && <AnswerCard answer={result.answer} citations={result.citations} />}
        </div>
      </div>
    </div>
  );
}

// =====================================================
// Compare
// =====================================================
function ComparePanel({ role }: { role: Role }) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{ answer: string; citations: Citation[] } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!a.trim() || !b.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await apiCompare(a.trim(), b.trim(), role);
      setResult(r);
    } catch (e) {
      setError(handleApiError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Compare" subtitle="Compare two regulatory topics side-by-side." />
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-5">
          <div className="rounded-lg border border-border bg-card p-5">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium mb-2">Topic A</label>
                <Textarea
                  value={a}
                  onChange={(e) => setA(e.target.value)}
                  placeholder="e.g. MahaRERA project registration fees"
                  className="min-h-[80px]"
                />
              </div>
              <div>
                <label className="block text-sm font-medium mb-2">Topic B</label>
                <Textarea
                  value={b}
                  onChange={(e) => setB(e.target.value)}
                  placeholder="e.g. RERA national act registration fees"
                  className="min-h-[80px]"
                />
              </div>
            </div>
            <div className="mt-4 flex justify-end">
              <Button onClick={run} disabled={busy || !a.trim() || !b.trim()}>
                {busy ? <Loader2 className="h-4 w-4 animate-spin" /> : "Compare"}
              </Button>
            </div>
          </div>
          {error && <ErrorCard message={error} />}
          {result && <AnswerCard answer={result.answer} citations={result.citations} />}
        </div>
      </div>
    </div>
  );
}

// =====================================================
// Search
// =====================================================
const CATEGORY_COLOR: Record<string, string> = {
  acts: "bg-primary/15 text-primary",
  act: "bg-primary/15 text-primary",
  circulars: "bg-emerald-500/15 text-emerald-400",
  circular: "bg-emerald-500/15 text-emerald-400",
  regulations: "bg-violet-500/15 text-violet-400",
  regulation: "bg-violet-500/15 text-violet-400",
  rules: "bg-yellow-500/15 text-yellow-400",
  rule: "bg-yellow-500/15 text-yellow-400",
};

function SearchPanel({ role }: { role: Role }) {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<"bm25" | "fulltext" | "metadata">("bm25");
  const [busy, setBusy] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [error, setError] = useState<string | null>(null);

  const run = async (qq: string = q, mm: typeof mode = mode) => {
    if (!qq.trim() || busy) return;
    setBusy(true);
    setError(null);
    try {
      const r = await apiSearch(qq.trim(), mm, role);
      setResults(r);
    } catch (e) {
      setError(handleApiError(e));
      setResults([]);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <PageHeader title="Search" subtitle="Full-text and BM25 search across the document corpus." />
      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-5">
          <div className="rounded-lg border border-border bg-card p-4">
            <div className="flex gap-2">
              <Input
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && run()}
                placeholder="Search documents…"
              />
              <Button onClick={() => run()} disabled={busy || !q.trim()}>
                {busy ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <SearchIcon className="h-4 w-4" />
                )}
              </Button>
            </div>
            <div className="mt-3 flex gap-1 p-1 bg-muted/50 rounded-md w-fit">
              {(["bm25", "fulltext", "metadata"] as const).map((m) => (
                <button
                  key={m}
                  onClick={() => {
                    setMode(m);
                    if (q.trim()) run(q, m);
                  }}
                  className={cn(
                    "text-xs font-medium px-3 py-1 rounded transition-colors capitalize",
                    mode === m
                      ? "bg-card text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground",
                  )}
                >
                  {m === "bm25" ? "BM25" : m === "fulltext" ? "Full-text" : "Metadata"}
                </button>
              ))}
            </div>
          </div>

          {error && <ErrorCard message={error} />}

          {results.length === 0 && !busy && !error && (
            <EmptyState
              icon={SearchIcon}
              title="Search the document corpus"
              description="Use BM25 for keyword relevance, full-text for phrase matches, or metadata to filter by title."
            />
          )}

          <div className="space-y-2">
            {results.map((r) => {
              const cat = r.category?.toLowerCase() || "";
              const color = CATEGORY_COLOR[cat] || "bg-muted text-muted-foreground";
              return (
                <div
                  key={r.document_id}
                  className="rounded-lg border border-border bg-card p-4 hover:border-primary/40 transition-colors"
                >
                  <div className="flex items-start justify-between gap-3">
                    <h3 className="text-sm font-semibold text-foreground">{r.title}</h3>
                    {r.category && (
                      <span
                        className={cn(
                          "text-[10px] font-semibold uppercase px-2 py-0.5 rounded shrink-0",
                          color,
                        )}
                      >
                        {r.category}
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground mt-1.5 leading-relaxed line-clamp-3">
                    {r.snippet}
                  </p>
                  <div className="mt-3 flex items-center gap-2">
                    <div className="h-1 flex-1 rounded-full bg-muted overflow-hidden">
                      <div
                        className="h-full bg-primary rounded-full"
                        style={{ width: `${Math.min(100, Math.max(0, r.score * 100))}%` }}
                      />
                    </div>
                    <span className="text-[10px] font-medium text-muted-foreground tabular-nums">
                      {r.score.toFixed(2)}
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// =====================================================
// Agent
// =====================================================
const AGENT_TYPES = ["Document Analyst", "Comparison", "Research", "Compliance", "Report"];

const AGENT_COLORS: Record<string, string> = {
  document_analyst: "bg-primary/15 text-primary border-primary/30",
  comparison: "bg-violet-500/15 text-violet-300 border-violet-500/30",
  compliance: "bg-yellow-500/15 text-yellow-300 border-yellow-500/30",
  research: "bg-emerald-500/15 text-emerald-300 border-emerald-500/30",
  report: "bg-muted text-muted-foreground border-border",
};

function AgentPanel({ role }: { role: Role }) {
  const [task, setTask] = useState("");
  const [busy, setBusy] = useState(false);
  const [result, setResult] = useState<{
    answer: string;
    citations: Citation[];
    agent: string;
  } | null>(null);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!task.trim() || busy) return;
    setBusy(true);
    setError(null);
    setResult(null);
    try {
      const r = await runAgent(task.trim(), role);
      setResult(r);
    } catch (e) {
      setError(handleApiError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-border bg-card/50 px-8 py-5">
        <div className="flex items-center gap-2">
          <Zap className="h-5 w-5 text-violet-400" />
          <h1 className="text-xl font-semibold tracking-tight text-violet-300">AI Agent</h1>
          <Badge className="bg-violet-500/20 text-violet-300 border-violet-500/30 hover:bg-violet-500/20">
            Phase 5
          </Badge>
        </div>
        <p className="text-sm text-muted-foreground mt-1">
          Routes your task to the right specialist: Document Analyst, Comparison, Compliance,
          Research, or Report agent.
        </p>
      </div>

      <div className="flex-1 overflow-y-auto px-8 py-6">
        <div className="max-w-3xl mx-auto space-y-5">
          <div className="rounded-lg border border-violet-500/20 bg-card p-5">
            <label className="block text-sm font-medium mb-2">Describe your task</label>
            <Textarea
              value={task}
              onChange={(e) => setTask(e.target.value)}
              placeholder="e.g. Compare MahaRERA registration fees vs RERA national act, or Check if a builder cancelling a booking violates MahaRERA rules"
              className="min-h-[110px] focus-visible:ring-violet-500/50"
            />
            <div className="mt-3 flex flex-wrap gap-1.5">
              {AGENT_TYPES.map((t) => (
                <span
                  key={t}
                  className="text-[11px] px-2 py-0.5 rounded-full bg-violet-500/10 text-violet-300 border border-violet-500/20"
                >
                  {t}
                </span>
              ))}
            </div>
            <div className="mt-4 flex justify-end">
              <Button
                onClick={run}
                disabled={busy || !task.trim()}
                className="bg-violet-600 hover:bg-violet-700 text-white"
              >
                {busy ? (
                  <Loader2 className="h-4 w-4 animate-spin" />
                ) : (
                  <Zap className="h-4 w-4" />
                )}
                <span className="ml-2">Run agent</span>
              </Button>
            </div>
          </div>

          {error && <ErrorCard message={error} />}

          {result && (
            <div className="space-y-3">
              <div>
                <span
                  className={cn(
                    "inline-flex items-center gap-1.5 text-xs font-semibold px-2.5 py-1 rounded-full border",
                    AGENT_COLORS[result.agent] || "bg-muted text-muted-foreground border-border",
                  )}
                >
                  <Zap className="h-3 w-3" />
                  Handled by:{" "}
                  {result.agent
                    .split("_")
                    .map((s) => s.charAt(0).toUpperCase() + s.slice(1))
                    .join(" ")}{" "}
                  Agent
                </span>
              </div>
              <AnswerCard answer={result.answer} citations={result.citations || []} />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
