import { DefaultChatTransport } from "ai"

import { api } from "@/lib/api"
import type { ChatThread, CopilotMessage } from "@/lib/chat"
import { env } from "@/lib/env"
import { getAccessToken } from "@/lib/session"

export function createChatTransport(
  threadId: string | null,
  onThread?: (id: string) => void,
) {
  let resolvedId = threadId
  return new DefaultChatTransport<CopilotMessage>({
    api: `${env.apiBaseUrl}/chat/stream`,
    headers: async () => {
      const token = await getAccessToken()
      const headers: Record<string, string> = {}
      if (token) {
        headers.Authorization = `Bearer ${token}`
      }
      return headers
    },
    prepareSendMessagesRequest: async ({ messages }) => {
      if (!resolvedId) {
        const thread = await api.post<ChatThread>("/chat/threads", {})
        resolvedId = thread.id
        onThread?.(resolvedId)
      }
      return { body: { threadId: resolvedId, messages } }
    },
  })
}
