import { useEffect, useState } from "react";
import { BadgeCheck, Clock } from "lucide-react";
import { fmt, getJSON } from "../api";
import { Empty, PageTitle, Stat } from "../components/Bits";

type Filing = { filing_id: string; name: string; pdf_published_at: string; xbrl_published_at: string | null; split: string | null; answered: number; correct: number };
type Season = { filings: Filing[]; graded_episodes: number; correct: number; accuracy: number | null; run?: string | null };

function minutes(a: string, b: string | null) {
  if (!b) return null;
  return (new Date(b).getTime() - new Date(a).getTime()) / 60000;
}

export default function SeasonPage() {
  const [s, setS] = useState<Season | null>(null);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    getJSON<Season>("/season").then(setS).catch((e) => setErr(e.message));
  }, []);
  return (
    <div>
      <PageTitle title="Season" sub="Filings in the order they landed. The agent answers from the PDF the moment it appears, then grades itself once the exchange publishes the XBRL answer key." />
      {err && <Empty>{err}</Empty>}
      {s && (
        <>
          <div className="mb-4 grid grid-cols-2 gap-3 md:grid-cols-4">
            <Stat label="filings" value={s.filings.length} />
            <Stat label="graded answers" value={s.graded_episodes} />
            <Stat label="self graded accuracy" value={fmt.pct(s.accuracy)} hint={s.run ?? undefined} />
            <Stat label="graded correct" value={s.correct} />
          </div>
          <div className="panel overflow-x-auto p-2">
            <table className="datatable">
              <thead>
                <tr>
                  <th>company</th>
                  <th>PDF published (UTC)</th>
                  <th>XBRL lag</th>
                  <th>split</th>
                  <th>answers</th>
                  <th>status</th>
                </tr>
              </thead>
              <tbody>
                {s.filings.map((f) => {
                  const lag = minutes(f.pdf_published_at, f.xbrl_published_at);
                  return (
                    <tr key={f.filing_id}>
                      <td className="font-semibold">{f.name}</td>
                      <td className="num">{f.pdf_published_at.replace("T", " ").slice(0, 16)}</td>
                      <td className="num">{lag === null ? "waiting" : `${lag.toFixed(0)} min`}</td>
                      <td>{f.split ?? "-"}</td>
                      <td className="num">{f.answered ? `${f.correct}/${f.answered}` : "-"}</td>
                      <td>
                        {f.xbrl_published_at ? (
                          <span className="badge ok">
                            <BadgeCheck size={13} aria-hidden /> graded
                          </span>
                        ) : (
                          <span className="badge mute">
                            <Clock size={13} aria-hidden /> awaiting XBRL
                          </span>
                        )}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  );
}
