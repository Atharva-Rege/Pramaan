import { NavLink, Route, Routes } from "react-router-dom";
import { BadgeCheck, BarChart3, CalendarClock, ClipboardCheck, Info, MessageSquareText, Trophy } from "lucide-react";
import Ask from "./pages/Ask";
import Season from "./pages/Season";
import Lab from "./pages/Lab";
import Leaderboard from "./pages/Leaderboard";
import Review from "./pages/Review";
import About from "./pages/About";

const nav = [
  { to: "/", label: "Ask", icon: MessageSquareText },
  { to: "/season", label: "Season", icon: CalendarClock },
  { to: "/lab", label: "Lab", icon: BarChart3 },
  { to: "/leaderboard", label: "Leaderboard", icon: Trophy },
  { to: "/review", label: "Review", icon: ClipboardCheck },
  { to: "/about", label: "About", icon: Info },
];

export default function App() {
  return (
    <div className="flex min-h-full flex-col md:flex-row">
      <aside className="md:sticky md:top-0 md:h-screen md:w-52 shrink-0 border-b md:border-b-0 md:border-r border-rule bg-card">
        <div className="flex items-center gap-2 px-4 py-4">
          <span className="grid h-8 w-8 place-items-center rounded-lg bg-brand text-paper">
            <BadgeCheck size={18} aria-hidden />
          </span>
          <div>
            <div className="font-display text-xl leading-none">Pramaan</div>
            <div className="text-[11px] text-ink-soft">proof for every number</div>
          </div>
        </div>
        <nav className="flex gap-1 overflow-x-auto px-2 pb-2 md:flex-col md:px-3" aria-label="Main">
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              end={to === "/"}
              className={({ isActive }) =>
                `flex min-h-11 items-center gap-2 rounded-md px-3 text-sm whitespace-nowrap transition-colors ${
                  isActive ? "bg-brand-soft font-semibold text-brand" : "text-ink-soft hover:text-ink"
                }`
              }
            >
              <Icon size={16} aria-hidden />
              {label}
            </NavLink>
          ))}
        </nav>
        <p className="hidden px-4 pt-6 text-[11px] leading-snug text-ink-soft md:block">Research aid, not investment advice. Figures come from company filings.</p>
      </aside>
      <main className="min-w-0 flex-1 px-4 py-5 md:px-8">
        <Routes>
          <Route path="/" element={<Ask />} />
          <Route path="/season" element={<Season />} />
          <Route path="/lab" element={<Lab />} />
          <Route path="/leaderboard" element={<Leaderboard />} />
          <Route path="/review" element={<Review />} />
          <Route path="/about" element={<About />} />
        </Routes>
      </main>
    </div>
  );
}
