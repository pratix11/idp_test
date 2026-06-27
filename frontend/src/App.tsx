import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { toast } from "sonner";
import {
  Bot,
  Building2,
  GitCompareArrows,
  History,
  MessageSquare,
  Search,
  Send,
  Sparkles,
  Zap,
} from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import { Switch } from "@/components/ui/switch";
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import { Toaster } from "@/components/ui/sonner";
import { cn } from "@/lib/utils";
import {
  api,
  MOCK_ANSWER,
  MOCK_SEARCH,
  type AskResponse,
  type Citation,
  type SearchResult,
} from "@/lib/api";

type Tab = "chat" | "search" | "compare";

interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  streaming?: boolean;
}

export default function PropIntelApp() {
  const [tab, setTab] = useState<Tab>("chat");
  const [apiOk, setApiOk] = useState<boolean | null>(null);
  const [recent, setRecent] = useState<string[]>([]);
  const [pendingQuestion, setPendingQuestion] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      const ok = await api.health();
      if (!cancelled) setApiOk(ok);
    };
    check();
    const t = setInterval(check, 15000);
    return () => {
      cancelled = true;
      clearInterval(t);
    };
  }, []);

  const mockMode = apiOk === false;

  const addRecent = (q: string) => {
    setRecent((r) => [q, ...r.filter((x) => x !== q)].slice(0, 5));
  };

  return (
    <div className="flex h-screen w-full bg-background text-foreground overflow-hidden">
      <Toaster theme="dark" position="top-right" />
      {/* Sidebar */}
      <aside className="w-[260px] shrink-0 bg-sidebar border-r border-sidebar-border flex flex-col">
        <div className="px-5 py-5 flex items-center gap-3 border-b border-sidebar-border">
          <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-primary/40 to-primary/10 border border-primary/40 flex items-center justify-center shadow-lg shadow-primary/10">
            <Building2 className="w-5 h-5 text-primary" />
          </div>
          <div>
            <div className="font-bold tracking-tight text-sidebar-foreground text-base">PropIntel</div>
            <div className="text-[10px] text-primary/70 uppercase tracking-widest font-medium">
              AI Copilot
            </div>
          </div>
        </div>

        <nav className="px-3 py-4 space-y-1">
          <NavItem
            icon={<MessageSquare className="w-4 h-4" />}
            label="Chat"
            active={tab === "chat"}
            onClick={() => setTab("chat")}
          />
          <NavItem
            icon={<Search className="w-4 h-4" />}
            label="Search"
            active={tab === "search"}
            onClick={() => setTab("search")}
          />
          <NavItem
            icon={<GitCompareArrows className="w-4 h-4" />}
            label="Compare"
            active={tab === "compare"}
            onClick={() => setTab("compare")}
          />
        </nav>

        <Separator className="bg-sidebar-border" />

        <div className="px-3 py-4 flex-1 min-h-0 flex flex-col">
          <div className="px-2 mb-2 flex items-center gap-2 text-xs uppercase tracking-wider text-muted-foreground">
            <History className="w-3.5 h-3.5" /> Recent
          </div>
          <ScrollArea className="flex-1">
            {recent.length === 0 ? (
              <div className="px-2 text-xs text-muted-foreground/70 italic">
                Your recent questions will appear here.
              </div>
            ) : (
              <ul className="space-y-1">
                {recent.map((q, i) => (
                  <li key={i}>
                    <button
                      onClick={() => {
                        setTab("chat");
                        setPendingQuestion(q);
                      }}
                      className="w-full text-left px-2 py-1.5 rounded-md text-xs text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground line-clamp-2"
                    >
                      {q}
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </ScrollArea>
        </div>

        <div className="px-4 py-3 border-t border-sidebar-border flex items-center justify-between text-xs">
          <div className="flex items-center gap-2">
            <span
              className={cn(
                "w-2 h-2 rounded-full",
                apiOk === null && "bg-muted-foreground animate-pulse",
                apiOk === true && "bg-emerald-400 shadow-[0_0_8px] shadow-emerald-400/60",
                apiOk === false && "bg-rose-500",
              )}
            />
            <span className="text-muted-foreground">
              {apiOk === null ? "Checking…" : apiOk ? "API online" : "API offline"}
            </span>
          </div>
          {mockMode && (
            <Badge variant="outline" className="text-[10px] border-citation/40 text-citation">
              MOCK
            </Badge>
          )}
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 min-w-0 flex flex-col">
        {tab === "chat" && (
          <ChatPanel
            mockMode={mockMode}
            addRecent={addRecent}
            pendingQuestion={pendingQuestion}
            clearPending={() => setPendingQuestion(null)}
          />
        )}
        {tab === "search" && <SearchPanel mockMode={mockMode} />}
        {tab === "compare" && <ComparePanel mockMode={mockMode} addRecent={addRecent} />}
      </main>
    </div>
  );
}

function NavItem({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={cn(
        "w-full flex items-center gap-3 px-3 py-2 rounded-md text-sm transition-colors",
        active
          ? "bg-primary/15 text-primary border border-primary/25"
          : "text-sidebar-foreground/80 hover:bg-sidebar-accent hover:text-sidebar-accent-foreground border border-transparent",
      )}
    >
      {icon}
      <span className="font-medium">{label}</span>
    </button>
  );
}

/* ---------------- Chat ---------------- */

function ChatPanel({
  mockMode,
  addRecent,
  pendingQuestion,
  clearPending,
}: {
  mockMode: boolean;
  addRecent: (q: string) => void;
  pendingQuestion: string | null;
  clearPending: () => void;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [streaming, setStreaming] = useState(true);
  const [busy, setBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  useEffect(() => {
    if (pendingQuestion) {
      setInput(pendingQuestion);
      clearPending();
    }
  }, [pendingQuestion, clearPending]);

  const send = async () => {
    const q = input.trim();
    if (!q || busy) return;
    setInput("");
    setBusy(true);
    addRecent(q);

    const userMsg: ChatMessage = { id: `u-${Date.now()}`, role: "user", content: q };
    const asstId = `a-${Date.now()}`;
    setMessages((m) => [
      ...m,
      userMsg,
      { id: asstId, role: "assistant", content: "", streaming: true },
    ]);

    try {
      if (mockMode) {
        if (streaming) {
          const tokens = MOCK_ANSWER.answer.split(/(\s+)/);
          for (const t of tokens) {
            await new Promise((r) => setTimeout(r, 18));
            setMessages((m) =>
              m.map((msg) =>
                msg.id === asstId ? { ...msg, content: msg.content + t } : msg,
              ),
            );
          }
        }
        setMessages((m) =>
          m.map((msg) =>
            msg.id === asstId
              ? {
                  ...msg,
                  content: MOCK_ANSWER.answer,
                  citations: MOCK_ANSWER.citations,
                  streaming: false,
                }
              : msg,
          ),
        );
      } else if (streaming) {
        let acc = "";
        for await (const chunk of api.askStream(q)) {
          acc += chunk;
          setMessages((m) =>
            m.map((msg) => (msg.id === asstId ? { ...msg, content: acc } : msg)),
          );
        }
        setMessages((m) =>
          m.map((msg) =>
            msg.id === asstId ? { ...msg, streaming: false } : msg,
          ),
        );
      } else {
        const resp: AskResponse = await api.ask(q);
        setMessages((m) =>
          m.map((msg) =>
            msg.id === asstId
              ? { ...msg, content: resp.answer, citations: resp.citations, streaming: false }
              : msg,
          ),
        );
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      toast.error("Failed to get answer", { description: message });
      setMessages((m) =>
        m.map((msg) =>
          msg.id === asstId
            ? { ...msg, content: "_Error fetching answer._", streaming: false }
            : msg,
        ),
      );
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <PanelHeader
        icon={<MessageSquare className="w-4 h-4" />}
        title="Ask the Copilot"
        subtitle="Ask any question about Indian property regulations, MahaRERA, or the Real Estate Act."
      />

      <div className="px-6 pt-4 pb-2 border-b border-border bg-card/30">
        <div className="flex gap-2 items-end">
          <Textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="e.g. What are the registration requirements for real estate agents under MahaRERA?"
            className="min-h-[60px] resize-none bg-background border-border focus-visible:ring-primary/40"
            disabled={busy}
          />
          <Button
            onClick={send}
            disabled={busy || !input.trim()}
            size="lg"
            className="h-[60px] px-5"
          >
            <Send className="w-4 h-4 mr-2" />
            Ask
          </Button>
        </div>
        <div className="mt-2.5 flex items-center justify-between text-xs text-muted-foreground">
          <div className="flex items-center gap-2">
            <Zap className="w-3.5 h-3.5" />
            <span>Streaming</span>
            <Switch checked={streaming} onCheckedChange={setStreaming} />
          </div>
          <span className="opacity-70">Shift + Enter for newline</span>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 overflow-y-auto px-6 py-6 space-y-6">
        {messages.length === 0 && <EmptyChatHint onPick={(q) => setInput(q)} />}
        {messages.map((m) => (
          <MessageBubble key={m.id} msg={m} />
        ))}
      </div>
    </div>
  );
}

function EmptyChatHint({ onPick }: { onPick: (q: string) => void }) {
  const samples = [
    "What are the registration requirements for real estate agents under MahaRERA?",
    "Summarise the penalties for non-registration of a real estate project.",
    "What rights do allottees have if a project is delayed?",
  ];
  return (
    <div className="max-w-2xl mx-auto mt-10 text-center">
      <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-gradient-to-br from-primary/30 to-primary/5 border border-primary/30 mb-5 shadow-xl shadow-primary/10">
        <Sparkles className="w-8 h-8 text-primary" />
      </div>
      <h2 className="text-2xl font-bold tracking-tight">PropIntel Copilot</h2>
      <p className="text-sm text-muted-foreground mt-2 mb-7 max-w-md mx-auto leading-relaxed">
        Ask anything about MahaRERA regulations, the Real Estate Act, or related circulars.
        Get AI-generated answers with cited source documents.
      </p>
      <div className="grid gap-2">
        {samples.map((s) => (
          <button
            key={s}
            onClick={() => onPick(s)}
            className="text-left px-4 py-3 rounded-lg border border-border bg-card hover:border-primary/40 hover:bg-card/80 transition-colors text-sm"
          >
            {s}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ msg }: { msg: ChatMessage }) {
  if (msg.role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] rounded-2xl rounded-br-sm bg-gradient-to-br from-primary to-primary/80 text-primary-foreground px-4 py-3 text-sm shadow-md shadow-primary/20">
          {msg.content}
        </div>
      </div>
    );
  }
  return (
    <div className="flex justify-start gap-3">
      <div className="w-7 h-7 rounded-full bg-primary/20 border border-primary/40 flex items-center justify-center shrink-0 mt-0.5">
        <Bot className="w-3.5 h-3.5 text-primary" />
      </div>
      <div className="max-w-[85%] w-full">
        <div className="text-[11px] text-primary/60 font-medium mb-1.5 uppercase tracking-wider">PropIntel</div>
        <div className="prose prose-invert prose-sm max-w-none prose-p:my-2 prose-headings:mt-3 prose-headings:mb-2 prose-li:my-0.5 prose-strong:text-foreground text-foreground/90 leading-relaxed">
          <ReactMarkdown>{msg.content}</ReactMarkdown>
          {msg.streaming && (
            <span className="inline-block w-1.5 h-4 bg-primary/80 align-middle ml-0.5 animate-pulse rounded-sm" />
          )}
        </div>
        {msg.citations && msg.citations.length > 0 && (
          <CitationsBlock citations={msg.citations} />
        )}
      </div>
    </div>
  );
}

function CitationsBlock({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(true);
  return (
    <Collapsible open={open} onOpenChange={setOpen} className="mt-4">
      <CollapsibleTrigger asChild>
        <button className="flex items-center gap-2 text-xs font-medium text-citation hover:text-citation/80 transition-colors">
          <span className="w-1.5 h-1.5 rounded-full bg-citation" />
          {citations.length} {citations.length === 1 ? "Citation" : "Citations"}
          <span className="opacity-70">{open ? "▾" : "▸"}</span>
        </button>
      </CollapsibleTrigger>
      <CollapsibleContent className="mt-3 grid gap-2.5 sm:grid-cols-2">
        {citations.map((c) => (
          <CitationCard key={`${c.chunk_id}-${c.index}`} citation={c} />
        ))}
      </CollapsibleContent>
    </Collapsible>
  );
}

function CitationCard({ citation }: { citation: Citation }) {
  return (
    <Card className="relative p-3 pr-10 bg-card border-border hover:border-citation/40 transition-colors">
      <div className="absolute top-2 right-2 w-6 h-6 rounded-md bg-citation/15 border border-citation/40 flex items-center justify-center text-[11px] font-semibold text-citation">
        {citation.index}
      </div>
      <div className="text-xs font-medium text-foreground">
        {citation.document_title || `Document #${citation.document_id}`}
      </div>
      {citation.section_title && (
        <div className="text-[11px] text-muted-foreground mt-0.5">{citation.section_title}</div>
      )}
      <div className="text-xs text-foreground/75 mt-2 leading-snug">
        {citation.content_snippet.length > 200
          ? citation.content_snippet.slice(0, 200) + "…"
          : citation.content_snippet}
      </div>
    </Card>
  );
}

/* ---------------- Search ---------------- */

function SearchPanel({ mockMode }: { mockMode: boolean }) {
  const [q, setQ] = useState("");
  const [mode, setMode] = useState<"bm25" | "fulltext">("bm25");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [busy, setBusy] = useState(false);
  const [selected, setSelected] = useState<SearchResult | null>(null);

  const run = async () => {
    if (!q.trim()) return;
    setBusy(true);
    try {
      if (mockMode) {
        await new Promise((r) => setTimeout(r, 300));
        setResults(MOCK_SEARCH);
      } else {
        const r = await api.search(q.trim(), mode);
        setResults(r.results);
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      toast.error("Search failed", { description: message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <PanelHeader
        icon={<Search className="w-4 h-4" />}
        title="Document Search"
        subtitle="Search the indexed corpus of property regulations, acts, and circulars."
      />
      <div className="px-6 pt-4 pb-4 border-b border-border bg-card/30">
        <div className="flex gap-2">
          <Input
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && run()}
            placeholder="Search documents…"
            className="bg-background"
          />
          <div className="flex rounded-md border border-border overflow-hidden">
            {(["bm25", "fulltext"] as const).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={cn(
                  "px-3 text-xs font-medium transition-colors",
                  mode === m
                    ? "bg-primary text-primary-foreground"
                    : "bg-background text-muted-foreground hover:bg-muted",
                )}
              >
                {m === "bm25" ? "BM25" : "Full-text"}
              </button>
            ))}
          </div>
          <Button onClick={run} disabled={busy || !q.trim()}>
            <Search className="w-4 h-4 mr-1.5" /> Search
          </Button>
        </div>
      </div>

      <div className="flex-1 min-h-0 grid grid-cols-1 lg:grid-cols-[1fr_380px] divide-x divide-border">
        <ScrollArea className="h-full">
          <div className="p-6 space-y-3">
            {results.length === 0 && (
              <div className="text-sm text-muted-foreground text-center py-16">
                {busy ? "Searching…" : "No results yet. Try a query above."}
              </div>
            )}
            {results.map((r) => (
              <button
                key={r.document_id}
                onClick={() => setSelected(r)}
                className={cn(
                  "w-full text-left p-4 rounded-lg border bg-card hover:border-primary/40 transition-colors",
                  selected?.document_id === r.document_id ? "border-primary/60" : "border-border",
                )}
              >
                <div className="flex items-start justify-between gap-3">
                  <div className="font-medium text-sm">{r.title}</div>
                  <Badge variant="outline" className="text-[10px] border-citation/40 text-citation shrink-0">
                    {r.category}
                  </Badge>
                </div>
                <div className="mt-2 text-xs text-muted-foreground line-clamp-2">{r.snippet}</div>
                <div className="mt-3 flex items-center gap-2">
                  <div className="h-1 flex-1 rounded-full bg-muted overflow-hidden">
                    <div
                      className="h-full bg-primary"
                      style={{ width: `${Math.min(100, r.score * 100)}%` }}
                    />
                  </div>
                  <span className="text-[10px] text-muted-foreground tabular-nums">
                    {r.score.toFixed(2)}
                  </span>
                </div>
              </button>
            ))}
          </div>
        </ScrollArea>

        <div className="hidden lg:block bg-card/30">
          {selected ? (
            <div className="p-6">
              <Badge variant="outline" className="text-[10px] border-citation/40 text-citation mb-3">
                {selected.category}
              </Badge>
              <h3 className="text-lg font-semibold">{selected.title}</h3>
              <div className="mt-2 text-xs text-muted-foreground">
                Document ID: {selected.document_id} · Score {selected.score.toFixed(3)}
              </div>
              <Separator className="my-4" />
              <p className="text-sm text-foreground/85 leading-relaxed">{selected.snippet}</p>
            </div>
          ) : (
            <div className="h-full flex items-center justify-center text-xs text-muted-foreground p-6 text-center">
              Select a result to preview details.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

/* ---------------- Compare ---------------- */

function ComparePanel({
  mockMode,
  addRecent,
}: {
  mockMode: boolean;
  addRecent: (q: string) => void;
}) {
  const [a, setA] = useState("");
  const [b, setB] = useState("");
  const [busy, setBusy] = useState(false);
  const [resp, setResp] = useState<AskResponse | null>(null);

  const run = async () => {
    if (!a.trim() || !b.trim()) return;
    setBusy(true);
    addRecent(`Compare: ${a.trim()} vs ${b.trim()}`);
    try {
      if (mockMode) {
        await new Promise((r) => setTimeout(r, 500));
        setResp(MOCK_ANSWER);
      } else {
        setResp(await api.compare(a.trim(), b.trim()));
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Unknown error";
      toast.error("Compare failed", { description: message });
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="flex flex-col h-full">
      <PanelHeader
        icon={<GitCompareArrows className="w-4 h-4" />}
        title="Compare Topics"
        subtitle="Compare two regulatory topics, sections, or provisions side-by-side."
      />
      <div className="px-6 pt-4 pb-4 border-b border-border bg-card/30">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Topic A</label>
            <Input
              value={a}
              onChange={(e) => setA(e.target.value)}
              placeholder="e.g. MahaRERA agent registration"
              className="bg-background"
            />
          </div>
          <div>
            <label className="text-xs font-medium text-muted-foreground mb-1.5 block">Topic B</label>
            <Input
              value={b}
              onChange={(e) => setB(e.target.value)}
              placeholder="e.g. RERA project registration"
              className="bg-background"
            />
          </div>
        </div>
        <div className="mt-3 flex justify-end">
          <Button onClick={run} disabled={busy || !a.trim() || !b.trim()}>
            <GitCompareArrows className="w-4 h-4 mr-1.5" />
            {busy ? "Comparing…" : "Compare"}
          </Button>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <div className="px-6 py-6 max-w-3xl">
          {!resp && !busy && (
            <div className="text-sm text-muted-foreground text-center py-16">
              Enter two topics above to generate an AI comparison.
            </div>
          )}
          {busy && (
            <div className="text-sm text-muted-foreground text-center py-16 animate-pulse">
              Generating comparison…
            </div>
          )}
          {resp && (
            <>
              <div className="prose prose-invert prose-sm max-w-none prose-strong:text-foreground text-foreground/90 leading-relaxed">
                <ReactMarkdown>{resp.answer}</ReactMarkdown>
              </div>
              {resp.citations.length > 0 && <CitationsBlock citations={resp.citations} />}
            </>
          )}
        </div>
      </ScrollArea>
    </div>
  );
}

/* ---------------- Shared ---------------- */

function PanelHeader({
  icon,
  title,
  subtitle,
}: {
  icon: React.ReactNode;
  title: string;
  subtitle: string;
}) {
  return (
    <div className="px-6 py-4 border-b border-border flex items-center gap-3">
      <div className="w-8 h-8 rounded-md bg-primary/10 border border-primary/25 flex items-center justify-center text-primary">
        {icon}
      </div>
      <div>
        <h1 className="text-sm font-semibold tracking-tight">{title}</h1>
        <p className="text-xs text-muted-foreground">{subtitle}</p>
      </div>
    </div>
  );
}
