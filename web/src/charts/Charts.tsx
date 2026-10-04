import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis, LabelList } from "recharts";

const C = { a: "var(--chart-a)", b: "var(--chart-b)", c: "var(--chart-c)", d: "var(--chart-d)" };
const axis = { stroke: "var(--ink-soft)", fontSize: 12 };
const grid = "color-mix(in srgb, var(--rule) 70%, transparent)";

export function Pareto({ rows }: { rows: { config: string; cost: number; acc: number }[] }) {
  return (
    <div role="img" aria-label="Scatter of verified accuracy against cost per question for each configuration" style={{ height: 320 }}>
      <ResponsiveContainer>
        <ScatterChart margin={{ top: 12, right: 28, bottom: 28, left: 8 }}>
          <CartesianGrid stroke={grid} />
          <XAxis type="number" dataKey="cost" scale="log" domain={["auto", "auto"]} name="cost" {...axis} label={{ value: "cost per question (USD, log scale)", position: "bottom", offset: 10, fill: "var(--ink-soft)", fontSize: 12 }} tickFormatter={(v: number) => `$${v < 0.01 ? v.toExponential(0) : v.toFixed(2)}`} />
          <YAxis type="number" dataKey="acc" domain={[0, 100]} name="verified accuracy" unit="%" {...axis} />
          <ZAxis range={[90, 90]} />
          <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(v: any, n: any) => (n === "cost" ? `$${Number(v).toFixed(4)}` : `${Number(v).toFixed(1)}%`)} />
          <Scatter data={rows} fill={C.a}>
            <LabelList dataKey="config" position="right" style={{ fill: "var(--ink-soft)", fontSize: 11 }} />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}

export function RiskCoverage({ coverage, risk }: { coverage: number[]; risk: number[] }) {
  const data = coverage.map((c, i) => ({ coverage: c, risk: risk[i] * 100 }));
  return (
    <div role="img" aria-label="Error rate among answered questions as coverage increases" style={{ height: 280 }}>
      <ResponsiveContainer>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 24, left: 8 }}>
          <CartesianGrid stroke={grid} />
          <XAxis dataKey="coverage" type="number" domain={[0, 1]} {...axis} label={{ value: "coverage", position: "bottom", offset: 6, fill: "var(--ink-soft)", fontSize: 12 }} />
          <YAxis {...axis} unit="%" label={{ value: "error among answered", angle: -90, position: "insideLeft", fill: "var(--ink-soft)", fontSize: 12 }} />
          <Tooltip formatter={(v: any) => `${Number(v).toFixed(1)}%`} />
          <Line dataKey="risk" stroke={C.a} dot={false} strokeWidth={2.2} />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

export function Attribution({ before, after, flips }: { before: Record<string, number>; after?: Record<string, number>; flips?: Record<string, number | null> }) {
  const stages = Object.keys(before);
  const data = stages.map((s) => ({ stage: s, before: before[s] * 100, after: after ? after[s] * 100 : undefined, fixed: flips && flips[s] != null ? (flips[s] as number) * 100 : undefined }));
  return (
    <div className="grid gap-4 md:grid-cols-2">
      <div role="img" aria-label="Share of failures by stage" style={{ height: 260 }}>
        <ResponsiveContainer>
          <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
            <CartesianGrid stroke={grid} vertical={false} />
            <XAxis dataKey="stage" {...axis} />
            <YAxis {...axis} unit="%" />
            <Tooltip formatter={(v: any) => `${Number(v).toFixed(1)}%`} />
            <Legend />
            <Bar dataKey="before" fill={C.b} name="before" radius={[3, 3, 0, 0]} />
            {after && <Bar dataKey="after" fill={C.a} name="after" radius={[3, 3, 0, 0]} />}
          </BarChart>
        </ResponsiveContainer>
      </div>
      {flips && (
        <div role="img" aria-label="Share of failed episodes fixed by injecting the gold artifact at each stage" style={{ height: 260 }}>
          <ResponsiveContainer>
            <BarChart data={data.filter((d) => d.fixed !== undefined)} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
              <CartesianGrid stroke={grid} vertical={false} />
              <XAxis dataKey="stage" {...axis} />
              <YAxis {...axis} unit="%" domain={[0, 100]} />
              <Tooltip formatter={(v: any) => `${Number(v).toFixed(1)}%`} />
              <Bar dataKey="fixed" fill={C.c} name="fixed by gold injection" radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

export function LagHistogram({ edges, counts }: { edges: number[]; counts: number[] }) {
  const label = (a: number, b: number) => (a < -1e8 ? `<${b}` : b > 1e8 ? `>${a}` : `${a} to ${b}`);
  const data = counts.map((c, i) => ({ bin: label(edges[i], edges[i + 1]), filings: c }));
  return (
    <div role="img" aria-label="Histogram of minutes between the results PDF and the XBRL filing" style={{ height: 260 }}>
      <ResponsiveContainer>
        <BarChart data={data} margin={{ top: 8, right: 8, bottom: 40, left: 0 }}>
          <CartesianGrid stroke={grid} vertical={false} />
          <XAxis dataKey="bin" {...axis} interval={0} angle={-30} textAnchor="end" height={60} />
          <YAxis {...axis} allowDecimals={false} />
          <Tooltip />
          <Bar dataKey="filings" fill={C.a} radius={[3, 3, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
