import { useEffect, useState } from "react";
import { fmt, getJSON } from "../api";
import { Empty, PageTitle } from "../components/Bits";

type Row = Record<string, any>;

export default function Leaderboard() {
  const [data, setData] = useState<Record<string, Row[]> | null>(null);
  const [split, setSplit] = useState("test");
  useEffect(() => {
    getJSON<Record<string, Row[]>>("/leaderboard").then(setData).catch(() => setData(null));
  }, []);
  const rows = (data?.[split] ?? []).slice().sort((a, b) => b.verified_accuracy - a.verified_accuracy);
  return (
    <div>
      <PageTitle title="Leaderboard" sub="All configurations and baselines on the same questions. Verified accuracy needs a correct answer and citations that check out. Ranges are 95% intervals clustered by filing." />
      <div className="mb-3 flex gap-2" role="tablist">
        {["dev", "test"].map((s) => (
          <button key={s} role="tab" aria-selected={s === split} className="chip" style={s === split ? { borderColor: "var(--brand)", background: "var(--brand-soft)", color: "var(--ink)" } : undefined} onClick={() => setSplit(s)}>
            {s}
          </button>
        ))}
      </div>
      {rows.length === 0 ? (
        <Empty>No results for the {split} split yet.</Empty>
      ) : (
        <div className="panel overflow-x-auto p-2">
          <table className="datatable">
            <thead>
              <tr>
                <th>config</th>
                <th>n</th>
                <th>accuracy</th>
                <th>verified accuracy</th>
                <th>abstains</th>
                <th>$ per question</th>
                <th>$ per verified answer</th>
                <th>p50</th>
                <th>p95</th>
                <th>source</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r) => (
                <tr key={r.config}>
                  <td className="font-semibold">{r.config}</td>
                  <td className="num">{r.n}</td>
                  <td className="num">{fmt.pct(r.accuracy)}</td>
                  <td className="num">
                    {fmt.pct(r.verified_accuracy)} <span className="text-xs text-ink-soft">({(100 * r.verified_accuracy_ci[0]).toFixed(0)} to {(100 * r.verified_accuracy_ci[1]).toFixed(0)})</span>
                  </td>
                  <td className="num">{fmt.pct(r.abstain_rate)}</td>
                  <td className="num">{fmt.usd(r.cost_per_question_usd)}</td>
                  <td className="num">{fmt.usd(r.cost_per_verified_correct_usd)}</td>
                  <td className="num">{fmt.ms(r.latency_ms_p50)}</td>
                  <td className="num">{fmt.ms(r.latency_ms_p95)}</td>
                  <td>{r.provenance === "offline-heuristic" ? <span className="badge mute">rules only</span> : <span className="badge ok">LLM</span>}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
