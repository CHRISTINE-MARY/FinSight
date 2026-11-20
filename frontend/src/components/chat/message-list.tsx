import { PipelineStatusList } from "@/components/chat/pipeline-status"
import {
  citationLabel,
  citationsOf,
  lacksEvidence,
  messageText,
  statusesOf,
  type CitationSelection,
  type CopilotMessage,
} from "@/lib/chat"
import { cn } from "cn"

export function MessageList({
  messages,
  streaming,
  selection,
  onSelectCitation,
}: {
  messages: CopilotMessage[]
  streaming: boolean
  selection: CitationSelection | null
  onSelectCitation: (selection: CitationSelection) => void
}) {
  const liveId = streaming ? messages.at(-1)?.id : undefined

  return (
    <div className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto px-4 py-6">
      {messages.length === 0 ? (
        <div className="m-auto max-w-sm text-center">
          <p className="text-sm font-medium">No messages yet</p>
          <p className="mt-1 text-sm text-muted-foreground">
            Ask about a company, a filing, or a risk factor. Each claim links to the
            passage it came from.
          </p>
        </div>
      ) : (
        messages.map((message) => {
          const text = messageText(message)
          const citations = citationsOf(message)
          const noMatch = message.role === "assistant" && citations.length === 0 && lacksEvidence(text)
          return (
            <article
              key={message.id}
              className={
                message.role === "user"
                  ? "ml-auto max-w-xl rounded-xl bg-muted px-3 py-2 text-sm"
                  : "mr-auto w-full max-w-2xl text-sm"
              }
            >
              <p className="mb-1 text-xs text-muted-foreground">
                {message.role === "user" ? "You" : "Copilot"}
              </p>
              <p className="whitespace-pre-wrap">{text}</p>
              {noMatch ? (
                <p className="mt-3 rounded-lg border border-dashed px-3 py-2 text-muted-foreground">
                  No matching passage in the corpus.
                </p>
              ) : null}
              {citations.length > 0 ? (
                <div className="mt-3 flex flex-wrap gap-2">
                  {citations.map((citation, index) => {
                    const selected =
                      selection?.messageId === message.id && selection.index === index
                    return (
                      <button
                        key={`${citation.chunkId}-${index}`}
                        type="button"
                        aria-pressed={selected}
                        onClick={() => onSelectCitation({ messageId: message.id, index })}
                        className={cn(
                          "rounded-lg border px-2.5 py-1 text-left text-xs transition-colors duration-150 ease-out active:scale-[0.98]",
                          selected
                            ? "border-foreground bg-foreground text-background"
                            : "border-border hover:bg-muted",
                        )}
                      >
                        {citationLabel(citation)}
                      </button>
                    )
                  })}
                </div>
              ) : null}
              {message.role === "assistant" ? (
                <PipelineStatusList
                  statuses={statusesOf(message)}
                  live={message.id === liveId}
                />
              ) : null}
            </article>
          )
        })
      )}
    </div>
  )
}
