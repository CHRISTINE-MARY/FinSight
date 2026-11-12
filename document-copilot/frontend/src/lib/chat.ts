import type { UIMessage } from "ai"

export type ChatThread = {
  id: string
  title: string
  createdAt: string
  updatedAt: string
}

export type StoredChatMessage = {
  id: string
  role: "user" | "assistant"
  parts: { type: string; text?: string }[]
}

export type ThreadDetail = ChatThread & {
  messages: StoredChatMessage[]
}

export function toUiMessages(messages: StoredChatMessage[]): UIMessage[] {
  return messages.map((message) => ({
    id: message.id,
    role: message.role,
    parts: message.parts.flatMap((part) =>
      part.type === "text" && typeof part.text === "string"
        ? [{ type: "text" as const, text: part.text }]
        : [],
    ),
  }))
}

export function messageText(message: UIMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => ("text" in part ? part.text : ""))
    .join("")
}
