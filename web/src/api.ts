export type Citation = {
  id: string;
  filing_id: string;
  page_id: string;
  quote: string;
  line_item?: string | null;
  period?: string | null;
  scope?: string | null;
  bbox?: number[] | null;
};

export type Answer = {
  answer_type: "number" | "ranking" | "unavailable" | "premise_false";
  value?: number | null;
  unit?: string | null;
  ranking: string[];
  inputs: { name: string; value: number; citation_id?: string | null }[];
  computation?: string | null;
  citations: Citation[];
  confidence: number;
  explanation: string;
  reason?: string | null;
};

export type Check = { id: string; status: "pass" | "soft_fail" | "hard_fail" | "skip"; message: string };
export type VerifyReport = { status: string; checks: Check[] };

export type TraceItem = { kind: string; data: Record<string, any>; at: number };

export type DemoQuestion = { qid: string; tier: string; question: string; as_of: string };

const BASE = "/api";

export async function getJSON<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(BASE + path, init);
  if (!r.ok) {
    let detail = "";
    try {
      detail = (await r.json()).detail ?? "";
    } catch {
      detail = r.statusText;
    }
    throw new Error(detail || `HTTP ${r.status}`);
  }
  return r.json() as Promise<T>;
}

export async function streamAsk(question: string, onEvent: (kind: string, data: any) => void, signal?: AbortSignal): Promise<void> {
  const r = await fetch(BASE + "/ask", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
    signal,
  });
  if (!r.ok || !r.body) {
    let detail = "";
    try {
      detail = (await r.json()).detail ?? "";
    } catch {
      detail = r.statusText;
    }
    throw new Error(detail || `HTTP ${r.status}`);
  }
  const reader = r.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let idx: number;
    while ((idx = buf.indexOf("\n\n")) >= 0) {
      const block = buf.slice(0, idx);
      buf = buf.slice(idx + 2);
      let kind = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) kind = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (data) {
        try {
          onEvent(kind, JSON.parse(data));
        } catch {
          onEvent(kind, { raw: data });
        }
      }
    }
  }
}

export function pageImageUrl(pageId: string, bbox?: number[] | null): string {
  const q = bbox && bbox.length === 4 ? `?box=${bbox.map((v) => v.toFixed(1)).join(",")}` : "";
  return `${BASE}/pages/${encodeURIComponent(pageId)}/image${q}`;
}

export const fmt = {
  num: (v: number | null | undefined, d = 2) =>
    v === null || v === undefined || Number.isNaN(v) ? "n/a" : v.toLocaleString("en-IN", { minimumFractionDigits: d, maximumFractionDigits: d }),
  pct: (v: number | null | undefined, d = 1) => (v === null || v === undefined ? "n/a" : `${(100 * v).toFixed(d)}%`),
  usd: (v: number | null | undefined, d = 4) => (v === null || v === undefined ? "n/a" : `$${v.toFixed(d)}`),
  ms: (v: number | null | undefined) => (v === null || v === undefined ? "n/a" : v >= 1000 ? `${(v / 1000).toFixed(1)} s` : `${v.toFixed(0)} ms`),
};

export function unitLabel(unit?: string | null): string {
  if (unit === "INR_crore") return "₹ crore";
  if (unit === "INR_per_share") return "₹ per share";
  if (unit === "percent") return "%";
  return "";
}
