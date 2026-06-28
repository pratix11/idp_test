import { toast } from "sonner";

// Empty BASE_URL: Vite proxy routes /api and /health → localhost:8000
const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

export type Role = "admin" | "analyst" | "viewer" | "auditor";

export interface Citation {
  index: number;
  chunk_id: number;
  document_id: number;
  document_title: string | null;
  section_title: string | null;
  content_snippet: string;
}

export interface AskResponse {
  answer: string;
  citations: Citation[];
}

export interface SearchResult {
  document_id: number;
  title: string;
  snippet: string;
  score: number;
  category: string;
}

export interface AgentResponse extends AskResponse {
  agent: string;
}

export interface MeResponse {
  user_id: string;
  role: string;
  permissions: string[];
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

function h(role: Role): HeadersInit {
  return { "Content-Type": "application/json", "X-User-Role": role };
}

async function handle<T>(res: Response): Promise<T> {
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j.detail || j.message || detail;
    } catch {
      /* ignore */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${BASE_URL}/health`, {
      signal: AbortSignal.timeout(3000),
    });
    if (!res.ok) return false;
    const j = await res.json();
    return j?.status === "ok";
  } catch {
    return false;
  }
}

export async function getMe(role: Role): Promise<MeResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/me`, { headers: h(role) });
  return handle<MeResponse>(res);
}

export async function ask(question: string, role: Role): Promise<AskResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/ask`, {
    method: "POST",
    headers: h(role),
    body: JSON.stringify({ question }),
  });
  return handle<AskResponse>(res);
}

export async function summarize(query: string, role: Role): Promise<AskResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/summarize`, {
    method: "POST",
    headers: h(role),
    body: JSON.stringify({ query }),
  });
  return handle<AskResponse>(res);
}

export async function compare(
  query_a: string,
  query_b: string,
  role: Role,
): Promise<AskResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/compare`, {
    method: "POST",
    headers: h(role),
    body: JSON.stringify({ query_a, query_b }),
  });
  return handle<AskResponse>(res);
}

export async function search(
  q: string,
  mode: "bm25" | "fulltext" | "metadata",
  role: Role,
  limit = 10,
): Promise<SearchResult[]> {
  const url = `${BASE_URL}/api/v1/search?q=${encodeURIComponent(q)}&mode=${mode}&limit=${limit}`;
  const res = await fetch(url, { headers: { "X-User-Role": role } });
  const j = await handle<{ results: SearchResult[] }>(res);
  return j.results ?? [];
}

export async function runAgent(task: string, role: Role): Promise<AgentResponse> {
  const res = await fetch(`${BASE_URL}/api/v1/agent`, {
    method: "POST",
    headers: h(role),
    body: JSON.stringify({ task }),
  });
  return handle<AgentResponse>(res);
}

export async function* askStream(question: string, role: Role): AsyncGenerator<string> {
  const res = await fetch(`${BASE_URL}/api/v1/ask/stream`, {
    method: "POST",
    headers: h(role),
    body: JSON.stringify({ question }),
  });
  if (!res.ok || !res.body) throw new ApiError(res.status, res.statusText);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    const lines = buf.split("\n");
    buf = lines.pop() || "";
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data:")) continue;
      const payload = trimmed.slice(5).trim();
      if (payload === "[DONE]") return;
      try {
        const parsed = JSON.parse(payload);
        if (parsed.text) yield parsed.text as string;
      } catch {
        /* ignore */
      }
    }
  }
}

export function handleApiError(e: unknown, fallback = "Request failed"): string {
  if (e instanceof ApiError) {
    if (e.status === 403)
      return "Your role doesn't have permission for this feature. Switch to analyst or admin.";
    if (e.status === 401) return "Unknown role";
    return `${e.status}: ${e.message}`;
  }
  const msg = e instanceof Error ? e.message : fallback;
  toast.error(msg);
  return msg;
}
