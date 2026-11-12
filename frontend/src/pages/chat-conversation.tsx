import { useMemo, useState } from "react"
import { useChat } from "@ai-sdk/react"
import type { UIMessage } from "ai"

import { Composer } from "@/components/chat/composer"
import { MessageList } from "@/components/chat/message-list"
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
  initialMessages: UIMessage[]
  onFinish: () => void
  onThread?: (id: string) => void
}) {
  const transport = useMemo(
    () => createChatTransport(threadId, onThread),
    [threadId, onThread],
  )
  const { messages, sendMessage, status, error } = useChat({
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
  const streaming = status === "submitted" || status === "streaming"

  return (
    <section className="flex min-h-0 flex-1 flex-col">
      <MessageList messages={messages} streaming={streaming} />
      {error || sendError ? (
        <p role="alert" className="px-4 text-sm text-destructive">
          {error?.message || sendError}
        </p>
      ) : null}
      <Composer
        pending={streaming}
        onSend={async (text) => {
          setSendError(null)
          try {
            await sendMessage({ text })
          } catch (caught) {
            setSendError(caught instanceof Error ? caught.message : "Could not send")
          }
        }}
      />
    </section>
  )
}
