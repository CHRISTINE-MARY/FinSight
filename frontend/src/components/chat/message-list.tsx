import type { UIMessage } from "ai"

import { messageText } from "@/lib/chat"

export function MessageList({
  messages,
  streaming,
}: {
  messages: UIMessage[]
  streaming: boolean
}) {
  return (
    <div className="flex flex-1 flex-col gap-4 overflow-y-auto px-4 py-6">
      {messages.length === 0 ? (
        <p className="text-sm text-muted-foreground">
          Ask a question about the filings.
        </p>
      ) : (
        messages.map((message) => (
          <article
            key={message.id}
            className={
              message.role === "user"
                ? "ml-auto max-w-xl rounded-xl bg-muted px-3 py-2 text-sm"
                : "mr-auto max-w-xl text-sm"
            }
          >
            <p className="mb-1 text-xs text-muted-foreground">
              {message.role === "user" ? "You" : "Copilot"}
            </p>
            <p className="whitespace-pre-wrap">{messageText(message)}</p>
          </article>
        ))
      )}
      {streaming ? (
        <p className="text-sm text-muted-foreground" role="status">
          Assistant is writing…
        </p>
      ) : null}
    </div>
  )
}
