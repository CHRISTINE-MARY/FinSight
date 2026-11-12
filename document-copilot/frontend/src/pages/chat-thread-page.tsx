import { useEffect, useState } from "react"
import { useOutletContext, useParams } from "react-router-dom"

import { api } from "@/lib/api"
import { toUiMessages, type ThreadDetail } from "@/lib/chat"
import { ApiError } from "@/lib/http"
import { ChatConversation } from "@/pages/chat-conversation"
import type { ChatOutletContext } from "@/pages/chat-layout"
import type { UIMessage } from "ai"

export function ChatThreadPage() {
  const { threadId = "" } = useParams()
  const { refreshThreads } = useOutletContext<ChatOutletContext>()

  return (
    <ThreadHistory
      key={threadId}
      threadId={threadId}
      refreshThreads={refreshThreads}
    />
  )
}

function ThreadHistory({
  threadId,
  refreshThreads,
}: {
  threadId: string
  refreshThreads: () => Promise<void>
}) {
  const [messages, setMessages] = useState<UIMessage[] | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let active = true
    api
      .get<ThreadDetail>(`/chat/threads/${threadId}`)
      .then((thread) => {
        if (active) {
          setMessages(toUiMessages(thread.messages))
        }
      })
      .catch((caught: unknown) => {
        if (!active) {
          return
        }
        if (caught instanceof ApiError && caught.status === 403) {
          setError("You don't have access to this conversation.")
          return
        }
        setError(
          caught instanceof ApiError ? caught.message : "Could not load this chat",
        )
      })
    return () => {
      active = false
    }
  }, [threadId])

  if (error) {
    return (
      <main className="flex flex-1 items-center justify-center px-6">
        <p role="alert" className="text-sm text-destructive">
          {error}
        </p>
      </main>
    )
  }

  if (!messages) {
    return (
      <main className="flex flex-1 items-center justify-center px-6">
        <p className="text-sm text-muted-foreground">Loading conversation…</p>
      </main>
    )
  }

  return (
    <ChatConversation
      key={threadId}
      chatId={threadId}
      threadId={threadId}
      initialMessages={messages}
      onFinish={() => {
        void refreshThreads()
      }}
    />
  )
}
