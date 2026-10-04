import { ReactNode } from "react";

export function PageTitle({ title, sub }: { title: string; sub?: string }) {
  return (
    <header className="mb-5">
      <h1 className="text-3xl">{title}</h1>
      {sub && <p className="mt-1 max-w-3xl text-ink-soft">{sub}</p>}
    </header>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="panel p-3">
      <div className="text-xs uppercase tracking-wide text-ink-soft">{label}</div>
      <div className="num mt-1 text-2xl font-semibold">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-ink-soft">{hint}</div>}
    </div>
  );
}

export function Section({ title, why, children }: { title: string; why: string; children: ReactNode }) {
  return (
    <section className="panel mb-5 p-4 rise">
      <h2 className="text-xl">{title}</h2>
      <p className="mb-3 mt-1 max-w-3xl text-sm text-ink-soft">{why}</p>
      {children}
    </section>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="rounded-md border border-dashed border-rule p-4 text-sm text-ink-soft">{children}</div>;
}

export function StatusBadge({ status }: { status: string }) {
  const cls = status === "pass" ? "ok" : status === "skip" ? "mute" : "warn";
  const text = status === "pass" ? "pass" : status === "skip" ? "skip" : status === "soft_fail" ? "warning" : "fail";
  return <span className={`badge ${cls}`}>{text}</span>;
}
