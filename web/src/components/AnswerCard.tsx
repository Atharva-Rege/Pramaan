import { BadgeCheck, CircleHelp, OctagonAlert } from "lucide-react";
import { Answer, Citation, fmt, unitLabel } from "../api";

export default function AnswerCard({
  answer,
  verified,
  selected,
  onSelect,
}: {
  answer: Answer;
  verified: boolean;
  selected: string | null;
  onSelect: (c: Citation) => void;
}) {
  const abst = answer.answer_type === "unavailable";
  const prem = answer.answer_type === "premise_false";
  return (
    <article className="panel rise p-4">
      <div className="mb-2 flex items-center gap-2">
        {abst ? (
          <span className="badge warn">
            <CircleHelp size={13} aria-hidden /> abstained
          </span>
        ) : prem ? (
          <span className="badge warn">
            <OctagonAlert size={13} aria-hidden /> false premise
          </span>
        ) : verified ? (
          <span className="badge ok">
            <BadgeCheck size={13} aria-hidden /> verified
          </span>
        ) : (
          <span className="badge mute">unverified</span>
        )}
        <span className="text-xs text-ink-soft">confidence {fmt.pct(answer.confidence, 0)}</span>
      </div>
      {answer.answer_type === "number" && (
        <div className="flex items-baseline gap-2">
          <span className="num text-4xl font-semibold">{fmt.num(answer.value, answer.unit === "percent" ? 1 : 2)}</span>
          <span className="text-ink-soft">{unitLabel(answer.unit)}</span>
        </div>
      )}
      {answer.answer_type === "ranking" && (
        <ol className="list-decimal pl-5 text-lg">
          {answer.ranking.map((r) => (
            <li key={r}>{r}</li>
          ))}
        </ol>
      )}
      {(abst || prem) && <p className="text-lg">{answer.reason ?? answer.explanation}</p>}
      <p className="mt-2 text-sm text-ink-soft">{answer.explanation}</p>
      {answer.computation && (
        <p className="num mt-2 text-xs text-ink-soft">
          {answer.computation} with {answer.inputs.map((i) => `${i.name}=${fmt.num(i.value, 2)}`).join(", ")}
        </p>
      )}
      {answer.citations.length > 0 && (
        <div className="mt-3 border-t border-rule pt-3">
          <div className="mb-1 text-xs uppercase tracking-wide text-ink-soft">Evidence</div>
          <ul className="flex flex-col gap-1.5">
            {answer.citations.map((c) => (
              <li key={c.id}>
                <button
                  className={`flex min-h-11 w-full items-center justify-between gap-2 rounded-md border px-3 text-left text-sm transition-colors ${
                    selected === c.id ? "border-brand bg-brand-soft" : "border-rule hover:border-brand"
                  }`}
                  onClick={() => onSelect(c)}
                >
                  <span>
                    <span className="font-semibold">{c.line_item?.replace(/_/g, " ")}</span>
                    <span className="text-ink-soft"> {c.period}</span>
                  </span>
                  <span className="num">{c.quote}</span>
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}
    </article>
  );
}
