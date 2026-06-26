const BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

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

async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE_URL}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${res.status} ${res.statusText}`);
  return res.json() as Promise<T>;
}

export const api = {
  baseUrl: BASE_URL,
  health: async (): Promise<boolean> => {
    try {
      const res = await fetch(`${BASE_URL}/health`, { signal: AbortSignal.timeout(3000) });
      if (!res.ok) return false;
      const data = await res.json();
      return data.status === "ok";
    } catch {
      return false;
    }
  },
  ask: (question: string) => postJSON<AskResponse>("/api/v1/ask", { question }),
  summarize: (query: string) => postJSON<AskResponse>("/api/v1/summarize", { query }),
  compare: (query_a: string, query_b: string) =>
    postJSON<AskResponse>("/api/v1/compare", { query_a, query_b }),
  search: async (q: string, mode = "bm25", limit = 10) => {
    const res = await fetch(
      `${BASE_URL}/api/v1/search?q=${encodeURIComponent(q)}&mode=${mode}&limit=${limit}`,
    );
    if (!res.ok) throw new Error(`${res.status}`);
    return res.json() as Promise<{ results: SearchResult[] }>;
  },
  askStream: async function* (question: string): AsyncGenerator<string> {
    const res = await fetch(`${BASE_URL}/api/v1/ask/stream`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok || !res.body) throw new Error(`${res.status}`);
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
  },
};

export const MOCK_ANSWER: AskResponse = {
  answer:
    "Under the **Maharashtra Real Estate (Regulation and Development) Act**, all real estate agents facilitating the sale of properties in MahaRERA-registered projects must register with the Authority before conducting any transactions [1].\n\n**Key registration requirements:**\n\n1. Application via Form G with prescribed fee\n2. PAN card, address proof, and photographs\n3. Income tax returns for the last three years\n4. Details of any past convictions or pending litigation [2]\n\nRegistration is valid for **five years** and is renewable. Agents must display their MahaRERA registration number in all advertisements and communications [3].",
  citations: [
    {
      index: 1,
      chunk_id: 101,
      document_id: 1,
      document_title: "MahaRERA_Regulations_2017",
      section_title: "Chapter III — Registration of Real Estate Agents",
      content_snippet:
        "Every real estate agent who intends to facilitate the sale or purchase of any plot, apartment or building in a registered real estate project shall make an application to the Authority for registration in Form 'G' as prescribed under these regulations.",
    },
    {
      index: 2,
      chunk_id: 102,
      document_id: 1,
      document_title: "MahaRERA_Regulations_2017",
      section_title: "Section 9 — Documents Required",
      content_snippet:
        "The applicant shall furnish PAN card, address proof, recent photographs, income tax returns for the preceding three financial years, and a self-declaration regarding any past convictions or pending legal proceedings.",
    },
    {
      index: 3,
      chunk_id: 203,
      document_id: 2,
      document_title: "Real_Estate_Act_2016",
      section_title: "Section 10 — Functions of Real Estate Agents",
      content_snippet:
        "The registration granted under section 9 shall be valid for a period of five years and may be renewed thereafter. Every registered agent shall quote the registration number in all advertisements, prospectus and communications.",
    },
  ],
};

export const MOCK_SEARCH: SearchResult[] = [
  {
    document_id: 1,
    title: "MahaRERA Regulations 2017",
    snippet:
      "Regulations governing real estate agents, project registration, and consumer protections under the Maharashtra Real Estate Authority...",
    score: 0.94,
    category: "Regulation",
  },
  {
    document_id: 2,
    title: "Real Estate (Regulation and Development) Act 2016",
    snippet:
      "Central Act establishing the regulatory framework for real estate transactions, project registration, and dispute resolution across India...",
    score: 0.88,
    category: "Act",
  },
  {
    document_id: 3,
    title: "MahaRERA Circular No. 28/2022",
    snippet:
      "Clarifications on quarterly project progress reporting and updated penalty structure for non-compliant promoters...",
    score: 0.71,
    category: "Circular",
  },
];
