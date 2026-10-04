import { useEffect, useState } from "react";
import { ExternalLink } from "lucide-react";
import { Citation, getJSON, pageImageUrl } from "../api";

export default function PageViewer({ citation }: { citation: Citation | null }) {
  const [link, setLink] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);

  useEffect(() => {
    setFailed(false);
    setLink(null);
    if (!citation?.filing_id) return;
    getJSON<{ pdf_url?: string }>(`/filings/${citation.filing_id}`)
      .then((d) => setLink(d.pdf_url ?? null))
      .catch(() => setLink(null));
  }, [citation?.filing_id]);

  if (!citation) {
    return (
      <div className="panel grid min-h-64 place-items-center p-6 text-center text-sm text-ink-soft">
        Pick a citation to see the filing page with the printed number highlighted.
      </div>
    );
  }
  return (
    <div className="panel p-3">
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="text-sm">
          <span className="font-semibold">{citation.line_item?.replace(/_/g, " ") ?? "figure"}</span>
          <span className="text-ink-soft"> as printed: </span>
          <span className="num rounded bg-sun/40 px-1.5 py-0.5">{citation.quote}</span>
        </div>
        {link && (
          <a className="inline-flex min-h-11 items-center gap-1 text-sm text-brand underline-offset-2 hover:underline" href={link} target="_blank" rel="noreferrer">
            Original filing <ExternalLink size={14} aria-hidden />
          </a>
        )}
      </div>
      {failed ? (
        <div className="rounded-md border border-dashed border-rule p-4 text-sm text-ink-soft">
          The page image is not available in this deployment. The quote above is the exact text on page <span className="num">{citation.page_id}</span>.
        </div>
      ) : (
        <img
          key={citation.page_id + String(citation.bbox)}
          src={pageImageUrl(citation.page_id, citation.bbox)}
          alt={`Filing page ${citation.page_id} with ${citation.quote} highlighted`}
          className="w-full rounded border border-rule"
          onError={() => setFailed(true)}
        />
      )}
      <div className="num mt-2 text-xs text-ink-soft">{citation.page_id}</div>
    </div>
  );
}
