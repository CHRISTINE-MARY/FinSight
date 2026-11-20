import { Button } from "@/components/ui/button"
import { citationLabel, formatFilingDate, type CitationData } from "@/lib/chat"

export function SourcePanel({
  citation,
  onClose,
}: {
  citation: CitationData
  onClose: () => void
}) {
  const showChunk = Boolean(citation.content && citation.content !== citation.excerpt)
  const filed = formatFilingDate(citation.filingDate)

  return (
    <aside
      aria-label="Source passage"
      className="flex max-h-80 w-full shrink-0 flex-col border-t bg-muted/30 md:max-h-none md:w-96 md:border-t-0 md:border-l"
    >
      <div className="flex items-start justify-between gap-3 px-4 py-3">
        <div className="min-w-0">
          <p className="truncate text-sm font-medium">
            {citation.companyName || citation.ticker || "Filing passage"}
          </p>
          <p className="text-xs text-muted-foreground">{citationLabel(citation)}</p>
          {citation.accessionNumber ? (
            <p className="mt-1 text-xs text-muted-foreground">
              {citation.ticker} {citation.filingType}
              {citation.fiscalYear ? ` FY${citation.fiscalYear}` : ""}
              {filed ? ` · filed ${filed}` : ""}
              {" · "}
              {citation.accessionNumber}
            </p>
          ) : null}
        </div>
        <Button type="button" variant="ghost" size="sm" onClick={onClose}>
          Close
        </Button>
      </div>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 pb-4">
        <p className="text-xs font-medium text-muted-foreground">Cited passage</p>
        <blockquote className="mt-2 border-l-2 border-foreground/20 pl-3 text-sm whitespace-pre-wrap">
          {citation.excerpt}
        </blockquote>
        {showChunk ? (
          <>
            <p className="mt-4 text-xs font-medium text-muted-foreground">Full chunk</p>
            <p className="mt-2 text-sm whitespace-pre-wrap">{citation.content}</p>
          </>
        ) : null}
      </div>
    </aside>
  )
}
