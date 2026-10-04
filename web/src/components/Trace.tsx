import { useEffect, useRef } from "react";
import { ArrowUpRight, Brain, CircleCheck, CircleX, Compass, FileSearch, ListChecks, RotateCcw, ShieldCheck, Wrench } from "lucide-react";
import { TraceItem } from "../api";
import { StatusBadge } from "./Bits";

function Row({ icon, title, children }: { icon: React.ReactNode; title: string; children?: React.ReactNode }) {
  return (
    <li className="rise relative flex gap-3 pb-3">
      <span className="mt-0.5 grid h-6 w-6 shrink-0 place-items-center rounded-full border border-rule bg-card text-brand">{icon}</span>
      <div className="min-w-0 flex-1">
        <div className="text-sm font-semibold">{title}</div>
        {children && <div className="mt-0.5 text-[13px] text-ink-soft">{children}</div>}
      </div>
    </li>
  );
}

export default function Trace({ items, running }: { items: TraceItem[]; running: boolean }) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => {
    end.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [items.length]);
  if (!items.length && !running) {
    return <div className="panel p-4 text-sm text-ink-soft">The agent's steps appear here as they happen: intent, plan, tool calls, checks, and the answer.</div>;
  }
  return (
    <ol className="panel max-h-[70vh] overflow-y-auto p-4" aria-live="polite">
      {items.map((it, i) => {
        const d = it.data;
        switch (it.kind) {
          case "mode":
            return <Row key={i} icon={<Compass size={14} />} title={d.mode === "demo" ? "Replaying a cached episode" : `Live run (${d.config ?? "agent"})`} />;
          case "lessons":
            return (
              <Row key={i} icon={<Brain size={14} />} title="Validated lessons applied">
                {(d.lessons ?? []).join(" | ")}
              </Row>
            );
          case "intent":
            return (
              <Row key={i} icon={<Compass size={14} />} title="Understood the question">
                {d.error ? d.error : `${d.intent?.operation} on ${(d.intent?.items ?? []).join(", ") || d.intent?.metric || d.intent?.ratio_name || "metric"} for ${(d.intent?.periods ?? []).join(", ")}`}
              </Row>
            );
          case "plan":
            return (
              <Row key={i} icon={<ListChecks size={14} />} title={`Plan (${d.tier ?? "small"})`}>
                {(d.plan?.steps ?? []).map((s: any) => `${s.id}: ${s.tool}`).join("  ") || "fallback plan"}
              </Row>
            );
          case "tool_call":
            return (
              <Row key={i} icon={<Wrench size={14} />} title={`${d.tool}`}>
                <span className="num">{JSON.stringify(d.args).slice(0, 120)}</span>
              </Row>
            );
          case "search":
            return (
              <Row key={i} icon={<FileSearch size={14} />} title="Searched the filings">
                {d.query}
              </Row>
            );
          case "tool_result":
            return (
              <Row key={i} icon={<ArrowUpRight size={14} />} title={d.tool === "get_value" ? (d.found === false ? "Value not found" : "Read a value") : "Read a table"}>
                {d.found === false ? d.reason : d.quote ? <span className="num">{d.quote} on {d.page_id}</span> : `${d.rows ?? ""} rows from ${d.source ?? ""}`}
              </Row>
            );
          case "verify":
            return (
              <Row key={i} icon={<ShieldCheck size={14} />} title={`Checks: ${d.report?.status}`}>
                <div className="flex flex-wrap gap-1.5 pt-1">
                  {(d.report?.checks ?? [])
                    .filter((c: any) => c.status !== "skip")
                    .map((c: any) => (
                      <span key={c.id} title={c.message} className="inline-flex items-center gap-1">
                        <span className="num text-xs">{c.id}</span>
                        <StatusBadge status={c.status} />
                      </span>
                    ))}
                </div>
              </Row>
            );
          case "reflect":
            return <Row key={i} icon={<RotateCcw size={14} />} title="Reflected on the failure and revised the plan">{d.failures}</Row>;
          case "escalate":
            return <Row key={i} icon={<ArrowUpRight size={14} />} title={`Escalated to the ${d.to} model`}>{d.reason}</Row>;
          case "answer":
            return (
              <Row key={i} icon={d.abstained ? <CircleX size={14} /> : <CircleCheck size={14} />} title={d.abstained ? "Abstained" : "Answered"} />
            );
          case "error":
            return <Row key={i} icon={<CircleX size={14} />} title="Error">{d.message}</Row>;
          default:
            return null;
        }
      })}
      {running && <li className="pl-9 text-sm text-ink-soft">working...</li>}
      <div ref={end} />
    </ol>
  );
}
