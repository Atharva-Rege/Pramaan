import { PageTitle } from "../components/Bits";

export default function About() {
  return (
    <div className="max-w-3xl">
      <PageTitle title="About Pramaan" sub="Pramaan means proof. Every answer shows the page and the exact printed number behind it, and is graded against the exchange's own machine readable filing." />
      <div className="panel space-y-4 p-5 leading-relaxed">
        <section>
          <h2 className="text-xl">Method</h2>
          <p className="mt-1 text-ink-soft">
            Since April 2025 listed companies file the signed results PDF and, shortly after, the same numbers as Integrated Filing XBRL. The PDF arrives first, so an agent must read it; the XBRL that follows is a free, automatic answer key. Pramaan generates questions from that key, retrieves pages with a dense plus lexical cascade and a cross-encoder fine-tuned on the key's page labels, reads tables with typed tools, never lets a language model do arithmetic, and checks every answer (quote on page, recomputation, units, accounting identities, period) before showing it. Failures are diagnosed against the key stage by stage, and the key's page labels train the reranker.
          </p>
        </section>
        <section>
          <h2 className="text-xl">Data policy</h2>
          <p className="mt-1 text-ink-soft">
            Documents are public exchange filings fetched at one request every two seconds per host for personal and educational use. This site shows page renders it produced and links to the original document; it does not mirror full PDFs. Raw exchange files are not redistributed with the released question set.
          </p>
        </section>
        <section>
          <h2 className="text-xl">Not investment advice</h2>
          <p className="mt-1 text-ink-soft">Pramaan is a research aid. Figures can be wrong, filings can be revised, and nothing here is a recommendation to buy or sell any security.</p>
        </section>
      </div>
    </div>
  );
}
