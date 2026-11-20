import { useMemo, useState } from "react"
import { useChat } from "@ai-sdk/react"

import { Composer } from "@/components/chat/composer"
import { MessageList } from "@/components/chat/message-list"
import { SourcePanel } from "@/components/chat/source-panel"
import { describeChatError, describeRequestError } from "@/lib/chat-errors"
import {
  citationsOf,
  statusesOf,
  type CitationSelection,
  type CopilotMessage,
} from "@/lib/chat"
import { createChatTransport } from "@/lib/chat-transport"

export function ChatConversation({
  chatId,
  threadId,
  initialMessages,
  onFinish,
  onThread,
}: {
  chatId: string
  threadId: string | null
  initialMessages: CopilotMessage[]
  onFinish: () => void
  onThread?: (id: string) => void
}) {
  const transport = useMemo(
    () => createChatTransport(threadId, onThread),
    [threadId, onThread],
  )
  const { messages, sendMessage, status, error } = useChat<CopilotMessage>({
    id: chatId,
    messages: initialMessages,
    transport,
    onFinish: (event) => {
      if (event.isError || event.isAbort || event.isDisconnect) {
        return
      }
      onFinish()
    },
  })
  const [sendError, setSendError] = useState<string | null>(null)
  const [selection, setSelection] = useState<CitationSelection | null>(null)
  const streaming = status === "submitted" || status === "streaming"
  const selectedMessage = messages.find((message) => message.id === selection?.messageId)
  const selected =
    selection && selectedMessage ? citationsOf(selectedMessage)[selection.index] : undefined
  const liveMessage = streaming ? messages.at(-1) : undefined
  const liveStatuses =
    liveMessage && liveMessage.role === "assistant" ? statusesOf(liveMessage) : []
  const liveStatus = liveStatuses.at(-1)
  const alert = error ? describeChatError(error.message) : sendError

  function onSelectCitation(next: CitationSelection) {
    setSelection((current) =>
      current?.messageId === next.messageId && current.index === next.index ? null : next,
    )
  }

  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        <MessageList
          messages={messages}
          streaming={streaming}
          selection={selection}
          onSelectCitation={onSelectCitation}
        />
        {selected ? (
          <SourcePanel citation={selected} onClose={() => setSelection(null)} />
        ) : null}
      </div>
      {alert ? (
        <p role="alert" className="px-4 pt-3 text-sm text-destructive">
          {alert}
        </p>
      ) : null}
      {streaming ? (
        <p role="status" className="px-4 pt-3 text-xs text-muted-foreground">
          {liveStatus ? `${liveStatus.phase}: ${liveStatus.detail}` : "Starting the turn…"}
        </p>
      ) : null}
      <Composer
        pending={streaming}
        onSend={async (text) => {
          setSendError(null)
          try {
            await sendMessage({ text })
          } catch (caught) {
            setSendError(describeRequestError(caught, "Could not send"))
          }
        }}
      />
    </section>
  )
}
