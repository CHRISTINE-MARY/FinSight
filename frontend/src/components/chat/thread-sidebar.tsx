import { Link } from "react-router-dom"

import { Button } from "@/components/ui/button"
import type { ChatThread } from "@/lib/chat"
import { cn } from "cn"

export function ThreadSidebar({
  threads,
  activeThreadId,
  onCreate,
  onSignOut,
}: {
  threads: ChatThread[]
  activeThreadId?: string
  onCreate: () => void
  onSignOut: () => void
}) {
  return (
    <aside className="flex max-h-64 w-full shrink-0 flex-col border-b bg-muted/40 md:h-svh md:max-h-none md:w-72 md:border-r md:border-b-0">
      <div className="flex items-center justify-between gap-3 px-4 py-4">
        <p className="font-heading text-sm font-medium">Document Copilot</p>
        <Button type="button" size="sm" onClick={onCreate}>
          New chat
        </Button>
      </div>
      <nav className="flex min-h-0 flex-1 flex-col gap-1 overflow-y-auto px-2 pb-2" aria-label="Past conversations">
        {threads.length === 0 ? (
          <p className="px-2 py-3 text-sm text-muted-foreground">
            No conversations yet. Start a chat to search the filings.
          </p>
        ) : (
          threads.map((thread) => (
            <Link
              key={thread.id}
              to={`/chats/${thread.id}`}
              className={cn(
                "truncate rounded-lg px-2 py-2 text-sm hover:bg-muted",
                thread.id === activeThreadId && "bg-muted font-medium",
              )}
              aria-current={thread.id === activeThreadId ? "page" : undefined}
            >
              {thread.title}
            </Link>
          ))
        )}
      </nav>
      <div className="border-t px-4 py-3">
        <Button type="button" variant="outline" size="sm" onClick={onSignOut}>
          Sign out
        </Button>
      </div>
    </aside>
  )
}
