import { useCallback, useEffect, useState } from "react";
import { Check, X } from "lucide-react";
import { getJSON } from "../api";
import { Empty, PageTitle } from "../components/Bits";

type Item = { done?: boolean; question?: any; image?: string | null };

export default function Review() {
  const [token, setToken] = useState(() => sessionStorage.getItem("pramaan_review_token") ?? "");
  const [kind, setKind] = useState<"human" | "adjudicate">("human");
  const [item, setItem] = useState<Item | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [note, setNote] = useState("");

  const load = useCallback(async () => {
    setErr(null);
    try {
      setItem(await getJSON<Item>(`/review/next?kind=${kind}`, { headers: { "X-Review-Token": token } }));
    } catch (e: any) {
      setErr(e.message);
      setItem(null);
    }
  }, [kind, token]);

  useEffect(() => {
    if (token) load();
  }, [load, token]);

  const submit = useCallback(
    async (body: Record<string, unknown>) => {
      if (!item?.question) return;
      try {
        await getJSON("/review", { method: "POST", headers: { "Content-Type": "application/json", "X-Review-Token": token }, body: JSON.stringify({ qid: item.question.qid, note, ...body }) });
        setNote("");
        load();
      } catch (e: any) {
        setErr(e.message);
      }
    },
    [item, note, token, load],
  );

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "TEXTAREA" || (e.target as HTMLElement).tagName === "INPUT") return;
      if (kind === "human") {
        if (e.key === "1") submit({ question_ok: true, gold_ok: true });
        if (e.key === "2") submit({ question_ok: false, gold_ok: true });
        if (e.key === "3") submit({ question_ok: true, gold_ok: false });
      } else {
        if (e.key === "1") submit({ verdict: "xbrl_correct" });
        if (e.key === "2") submit({ verdict: "agent_correct" });
        if (e.key === "3") submit({ verdict: "both_wrong" });
      }
      if (e.key === "s") load();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [kind, submit, load]);

  const q = item?.question;
  return (
    <div>
      <PageTitle title="Review" sub="Check that questions are unambiguous and gold answers match the printed filing, and adjudicate where the agent disagrees with XBRL but cites a valid page." />
      <div className="panel mb-4 flex flex-wrap items-end gap-3 p-3">
        <label className="text-sm">
          <span className="mb-1 block font-semibold">Review token</span>
          <input
            type="password"
            value={token}
            onChange={(e) => {
              setToken(e.target.value);
              sessionStorage.setItem("pramaan_review_token", e.target.value);
            }}
            className="min-h-11 rounded-md border border-rule bg-paper px-3"
          />
        </label>
        {(["human", "adjudicate"] as const).map((k) => (
          <button key={k} className="chip" onClick={() => setKind(k)} style={k === kind ? { borderColor: "var(--brand)", background: "var(--brand-soft)", color: "var(--ink)" } : undefined}>
            {k === "human" ? "validate gym" : "adjudicate disagreements"}
          </button>
        ))}
      </div>
      {err && <Empty>{err}</Empty>}
      {item?.done && <Empty>Nothing left in this queue.</Empty>}
      {q && (
        <div className="grid gap-4 lg:grid-cols-2">
          <div className="panel p-4">
            <p className="text-lg">{q.text}</p>
            <dl className="num mt-3 grid grid-cols-[auto_1fr] gap-x-3 gap-y-1 text-sm">
              <dt className="text-ink-soft">tier</dt>
              <dd>{q.tier}</dd>
              <dt className="text-ink-soft">gold</dt>
              <dd>{q.gold.answer_type === "number" ? `${q.gold.value} ${q.gold.unit ?? ""}` : q.gold.answer_type === "ranking" ? q.gold.ranking.join(" > ") : q.gold.answer_type}</dd>
              <dt className="text-ink-soft">page</dt>
              <dd>{q.gold_pages?.[0]}</dd>
            </dl>
            <textarea value={note} onChange={(e) => setNote(e.target.value)} rows={2} placeholder="note (optional)" className="mt-3 w-full rounded-md border border-rule bg-paper p-2 text-sm" aria-label="Reviewer note" />
            <div className="mt-3 flex flex-wrap gap-2">
              {kind === "human" ? (
                <>
                  <button className="btn" onClick={() => submit({ question_ok: true, gold_ok: true })}>
                    <Check size={14} aria-hidden /> 1 both fine
                  </button>
                  <button className="btn ghost" onClick={() => submit({ question_ok: false, gold_ok: true })}>
                    <X size={14} aria-hidden /> 2 question unclear
                  </button>
                  <button className="btn ghost" onClick={() => submit({ question_ok: true, gold_ok: false })}>
                    <X size={14} aria-hidden /> 3 gold wrong
                  </button>
                </>
              ) : (
                <>
                  <button className="btn" onClick={() => submit({ verdict: "xbrl_correct" })}>1 XBRL correct</button>
                  <button className="btn ghost" onClick={() => submit({ verdict: "agent_correct" })}>2 agent correct</button>
                  <button className="btn ghost" onClick={() => submit({ verdict: "both_wrong" })}>3 both wrong</button>
                </>
              )}
              <button className="btn ghost" onClick={load}>
                s skip
              </button>
            </div>
          </div>
          <div className="panel p-3">
            {item?.image ? <img src={item.image.startsWith("/pages") ? "/api" + item.image : item.image} alt="Filing page for this question" className="w-full rounded border border-rule" onError={(e) => ((e.target as HTMLImageElement).style.display = "none")} /> : <Empty>No page image.</Empty>}
          </div>
        </div>
      )}
    </div>
  );
}
