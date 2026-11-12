import { useCallback, useRef } from "react"
import { useNavigate, useOutletContext } from "react-router-dom"

import { ChatConversation } from "@/pages/chat-conversation"
import type { ChatOutletContext } from "@/pages/chat-layout"

export function ChatEmpty() {
  const { draftKey, refreshThreads } = useOutletContext<ChatOutletContext>()
  return <DraftChat key={draftKey} refreshThreads={refreshThreads} />
}

function DraftChat({ refreshThreads }: { refreshThreads: () => Promise<void> }) {
  const navigate = useNavigate()
  const threadId = useRef<string | null>(null)
  const onThread = useCallback((id: string) => {
    threadId.current = id
  }, [])

  return (
    <ChatConversation
      chatId="draft"
      threadId={null}
      initialMessages={[]}
      onThread={onThread}
      onFinish={() => {
        const id = threadId.current
        if (!id) {
          return
        }
        void refreshThreads()
          .catch(() => undefined)
          .then(() => {
            navigate(`/chats/${id}`, { replace: true })
          })
      }}
    />
  )
}
