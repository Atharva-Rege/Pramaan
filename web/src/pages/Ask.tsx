import { useEffect, useRef, useState } from "react";
import { Send, Square } from "lucide-react";
import { Answer, Citation, DemoQuestion, TraceItem, getJSON, streamAsk } from "../api";
import { PageTitle } from "../components/Bits";
import Trace from "../components/Trace";
import AnswerCard from "../components/AnswerCard";
import PageViewer from "../components/PageViewer";

const FALLBACK = [
  "What was Tata Consultancy Services's consolidated revenue from operations for the quarter ended 30 June 2026? Answer in ₹ crore.",
  "By what percentage did Infosys's consolidated net profit change year on year in Q1 FY27?",
  "Rank Asian Paints, Berger Paints and Pidilite by revenue from operations for the quarter ended 30 June 2026, highest first.",
];

export default function Ask() {
  const [q, setQ] = useState("");
  const [running, setRunning] = useState(false);
  const [items, setItems] = useState<TraceItem[]>([]);
  const [answer, setAnswer] = useState<Answer | null>(null);
  const [verified, setVerified] = useState(false);
  const [sel, setSel] = useState<Citation | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [demo, setDemo] = useState<DemoQuestion[]>([]);
  const abort = useRef<AbortController | null>(null);

  useEffect(() => {
    getJSON<DemoQuestion[]>("/demo/questions").then(setDemo).catch(() => setDemo([]));
  }, []);

  async function ask(text: string) {
    if (!text.trim()) return;
    setRunning(true);
    setItems([]);
    setAnswer(null);
    setSel(null);
    setError(null);
    setVerified(false);
    abort.current = new AbortController();
    const t0 = performance.now();
    try {
      await streamAsk(
        text,
        (kind, data) => {
          if (kind === "done") return;
          setItems((prev) => [...prev, { kind, data, at: performance.now() - t0 }]);
          if (kind === "verify") setVerified(data.report?.status !== "hard_fail");
          if (kind === "answer") {
            const a = data.answer as Answer;
            setAnswer(a);
            if (a.citations?.length) setSel(a.citations[0]);
          }
          if (kind === "error") setError(data.message);
        },
        abort.current.signal,
      );
    } catch (e: any) {
      if (e.name !== "AbortError") setError(e.message);
    } finally {
      setRunning(false);
    }
  }

  const chips = demo.length ? demo.slice(0, 6).map((d) => d.question) : FALLBACK;

  return (
    <div>
      <PageTitle title="Ask a filing" sub="Questions about Indian listed companies, answered from their results filings. Every number shows the page it came from and is checked before you see it." />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.05fr)_minmax(0,0.9fr)_minmax(0,1.1fr)]">
        <div className="flex flex-col gap-4">
          <form
            className="panel p-3"
            onSubmit={(e) => {
              e.preventDefault();
              ask(q);
            }}
          >
            <label htmlFor="q" className="mb-1 block text-sm font-semibold">
              Your question
            </label>
            <textarea
              id="q"
              value={q}
              onChange={(e) => setQ(e.target.value)}
              rows={4}
              className="w-full resize-y rounded-md border border-rule bg-paper p-3 text-[15px]"
              placeholder="What was Bajaj Auto's consolidated profit before tax for the quarter ended 30 June 2026?"
            />
            <div className="mt-2 flex items-center justify-between gap-2">
              <div className="text-xs text-ink-soft">Cross-company ranking, growth, ratios, and false premises work too.</div>
              {running ? (
                <button type="button" className="btn ghost" onClick={() => abort.current?.abort()}>
                  <Square size={14} aria-hidden /> Stop
                </button>
              ) : (
                <button type="submit" className="btn" disabled={!q.trim()}>
                  <Send size={14} aria-hidden /> Ask
                </button>
              )}
            </div>
          </form>
          <div className="flex flex-wrap gap-2" aria-label="Example questions">
            {chips.map((c) => (
              <button key={c} className="chip text-left" onClick={() => setQ(c)} title={c}>
                {c.length > 74 ? c.slice(0, 72) + "..." : c}
              </button>
            ))}
          </div>
          {error && (
            <div role="alert" className="rounded-md border border-stamp/50 bg-stamp-soft p-3 text-sm text-stamp">
              {error}
            </div>
          )}
          {answer && <AnswerCard answer={answer} verified={verified} selected={sel?.id ?? null} onSelect={setSel} />}
        </div>
        <Trace items={items} running={running} />
        <PageViewer citation={sel} />
      </div>
    </div>
  );
}
