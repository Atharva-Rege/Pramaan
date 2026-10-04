import { useEffect, useState } from "react";
import { getJSON, fmt } from "../api";
import { Empty, PageTitle, Section, Stat } from "../components/Bits";
import { Attribution, LagHistogram, Pareto, RiskCoverage } from "../charts/Charts";

type Dict = Record<string, any>;

function useLab(name: string) {
  const [d, setD] = useState<Dict | null>(null);
  useEffect(() => {
    getJSON<Dict>(`/lab/${name}`).then(setD).catch(() => setD(null));
  }, [name]);
  return d;
}

const first = (d: Dict | null, prefix: string): Dict | null => {
  if (!d) return null;
  const k = Object.keys(d).find((x) => x.startsWith(prefix));
  return k ? d[k] : null;
};

function Retrieval({ e2 }: { e2: Dict | null }) {
  const data = first(e2, "retrieval");
  const [mode, setMode] = useState("scoped+scope");
  if (!data) return <Empty>Retrieval results will appear after `pramaan exp E2`.</Empty>;
  const modes = Object.keys(data.results);
  const rows = data.results[mode] ?? data.results[modes[0]];
  const best = (k: string) => Math.max(...rows.map((r: Dict) => r[k]));
  return (
    <>
      <div className="mb-2 flex flex-wrap gap-2">
        {modes.map((m) => (
          <button key={m} className="chip" aria-pressed={m === mode} style={m === mode ? { borderColor: "var(--brand)", color: "var(--ink)", background: "var(--brand-soft)" } : undefined} onClick={() => setMode(m)}>
            {m}
          </button>
        ))}
      </div>
      <div className="overflow-x-auto">
        <table className="datatable">
          <thead>
            <tr>
              <th>stages</th>
              {["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"].map((k) => (
                <th key={k}>{k}</th>
              ))}
              <th>latency</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r: Dict) => (
              <tr key={r.stages}>
                <td>{r.stages}</td>
                {["recall@1", "recall@3", "recall@5", "recall@10", "mrr", "ndcg@10"].map((k) => (
                  <td key={k} className="num" style={r[k] === best(k) ? { fontWeight: 700, color: "var(--brand)" } : undefined}>
                    {r[k].toFixed(3)}
                  </td>
                ))}
                <td className="num">{fmt.ms(r.latency_ms.total)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  );
}

export default function Lab() {
  const e1 = useLab("E1");
  const e2 = useLab("E2");
  const e4 = useLab("E4");
  const e6 = useLab("E6");
  const e8 = useLab("E8");
  const [lb, setLb] = useState<Dict | null>(null);
  useEffect(() => {
    getJSON<Dict>("/leaderboard").then(setLb).catch(() => setLb(null));
  }, []);
  const pareto = ((lb?.dev?.length ? lb.dev : lb?.test) ?? []).map((r: Dict) => ({ config: r.config, cost: Math.max(r.cost_per_question_usd, 1e-5), acc: r.verified_accuracy * 100 }));
  const gym = first(e1, "gym");
  const cal = first(e4, "calibration");
  const attr = first(e6, "attribution");
  const season = first(e8, "season");

  return (
    <div>
      <PageTitle title="Lab" sub="Every figure here is read from results files written by the experiment runners. Nothing is typed in by hand." />
      <Section title="Is the gym valid?" why="Gold answers come from exchange XBRL, and a question is only asked when the same number is printed in the PDF. We check how often the two agree and which XBRL definitions differ from the printed row.">
        {gym ? (
          <div className="grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="facts checked" value={gym.pdf_vs_xbrl_agreement.facts_checked} />
            <Stat label="PDF equals XBRL" value={fmt.pct(gym.pdf_vs_xbrl_agreement.agree_rate_when_read)} hint="when the row is read" />
            <Stat label="definition conflicts" value={gym.structural_conflicts.count} hint="removed from the gym" />
            <Stat label="human check" value={gym.human_validation.reviewed} hint={gym.human_validation.status.split(":")[0]} />
          </div>
        ) : (
          <Empty>Run `pramaan exp E1`.</Empty>
        )}
      </Section>
      <Section title="What does each retrieval stage buy?" why="Dense search finds the right page quickly; the cross-encoder fine-tuned on free XBRL labels decides between near identical pages such as standalone and consolidated statements.">
        <Retrieval e2={e2} />
      </Section>
      <Section title="Accuracy versus cost" why="The cascade tries the cheap model first and escalates when the checks fail. Whether that pays off is what this chart tests; read it with the leaderboard, which marks exploratory variants.">
        {pareto.length ? <Pareto rows={pareto} /> : <Empty>Run `pramaan exp E3`.</Empty>}
      </Section>
      <Section title="Does it know when it does not know?" why="Sorting answers by calibrated confidence, the error rate among the answers kept should fall as we answer fewer questions.">
        {cal ? (
          <>
            <RiskCoverage coverage={cal.risk_coverage.coverage} risk={cal.risk_coverage.risk} />
            <div className="mt-2 text-sm text-ink-soft">
              AURC <span className="num">{cal.aurc_calibrated_cv.toFixed(4)}</span>, selective accuracy at 90% coverage <span className="num">{fmt.pct(cal.selective_accuracy_at_90_coverage)}</span>, calibration error <span className="num">{cal.ece_cv.toFixed(3)}</span>.
            </div>
          </>
        ) : (
          <Empty>Run `pramaan calibrate`.</Empty>
        )}
      </Section>
      <Section title="Where does it fail, and what fixes it?" why="Each wrong answer is assigned to the stage that failed. Injecting the gold artifact at one stage and replaying shows which stage is worth improving.">
        {attr ? <Attribution before={attr.before.share} after={attr.after?.share} flips={attr.counterfactual?.flip_rate} /> : <Empty>Run `pramaan attrib`.</Empty>}
      </Section>
      <Section title="What did the season show?" why="Companies file the PDF first and the machine readable XBRL later. The gap is the window in which an agent must answer from the PDF alone.">
        {season && season.lag.histogram.length ? (
          <>
            <LagHistogram edges={season.lag.histogram_edges_minutes} counts={season.lag.histogram} />
            <div className="mt-2 text-sm text-ink-soft">
              Median lag <span className="num">{season.lag.all.median_minutes.toFixed(0)} min</span>; PDF first in <span className="num">{fmt.pct(season.lag.all.pdf_first_share)}</span> of filings.
            </div>
          </>
        ) : (
          <Empty>Run `pramaan exp E8`.</Empty>
        )}
      </Section>
    </div>
  );
}
