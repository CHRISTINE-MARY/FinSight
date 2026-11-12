import { useCallback, useEffect, useState } from "react"
import { Outlet, useNavigate, useParams } from "react-router-dom"

import { ThreadSidebar } from "@/components/chat/thread-sidebar"
import { api } from "@/lib/api"
import type { ChatThread } from "@/lib/chat"
import { ApiError } from "@/lib/http"
import { useAuth } from "@/lib/use-auth"

export type ChatOutletContext = {
  refreshThreads: () => Promise<void>
  draftKey: number
}

export function ChatLayout() {
  const { threadId } = useParams()
  const navigate = useNavigate()
  const { signOut, accountError } = useAuth()
  const [threads, setThreads] = useState<ChatThread[]>([])
  const [error, setError] = useState<string | null>(null)
  const [draftKey, setDraftKey] = useState(0)

  const refreshThreads = useCallback(async () => {
    const data = await api.get<{ threads: ChatThread[] }>("/chat/threads")
    setThreads(data.threads)
  }, [])

  useEffect(() => {
    let active = true
    api
      .get<{ threads: ChatThread[] }>("/chat/threads")
      .then((data) => {
        if (active) {
          setThreads(data.threads)
        }
      })
      .catch((caught: unknown) => {
        if (!active) {
          return
        }
        setError(caught instanceof ApiError ? caught.message : "Could not load chats")
      })
    return () => {
      active = false
    }
  }, [])

  function onCreate() {
    setError(null)
    setDraftKey((key) => key + 1)
    navigate("/chats")
  }

  return (
    <div className="flex h-svh flex-col md:flex-row">
      <ThreadSidebar
        threads={threads}
        activeThreadId={threadId}
        onCreate={onCreate}
        onSignOut={() => void signOut()}
      />
      <div className="flex min-h-0 min-w-0 flex-1 flex-col">
        {accountError || error ? (
          <p role="alert" className="px-4 pt-4 text-sm text-destructive">
            {accountError || error}
          </p>
        ) : null}
        <Outlet context={{ refreshThreads, draftKey } satisfies ChatOutletContext} />
      </div>
    </div>
  )
}
