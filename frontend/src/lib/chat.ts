import type { UIMessage } from "ai"

export type ChatThread = {
  id: string
  title: string
  createdAt: string
  updatedAt: string
}

export type CitationData = {
  chunkId: string
  ticker: string
  companyName: string
  filingType: string
  filingDate: string
  fiscalYear: number | null
  accessionNumber: string
  page: string | null
  section: string | null
  excerpt: string
  content?: string
}

export type PipelineStatus = {
  phase: string
  detail: string
}

export type CopilotMessage = UIMessage<
  unknown,
  {
    citation: CitationData
    status: PipelineStatus
  }
>

export type StoredPart = {
  type: string
  text?: string
  id?: string
  data?: unknown
}

export type StoredChatMessage = {
  id: string
  role: "user" | "assistant"
  parts: StoredPart[]
}

export type ThreadDetail = ChatThread & {
  messages: StoredChatMessage[]
}

export type CitationSelection = {
  messageId: string
  index: number
}

export function toUiMessages(messages: StoredChatMessage[]): CopilotMessage[] {
  return messages.map((message) => ({
    id: message.id,
    role: message.role,
    parts: message.parts.flatMap((part) => {
      const converted = toPart(part)
      return converted ? [converted] : []
    }),
  }))
}

export function messageText(message: CopilotMessage): string {
  return message.parts
    .filter((part) => part.type === "text")
    .map((part) => ("text" in part ? part.text : ""))
    .join("")
}

export function citationsOf(message: CopilotMessage): CitationData[] {
  const citations: CitationData[] = []
  for (const part of message.parts) {
    if (part.type !== "data-citation" || !("data" in part)) {
      continue
    }
    const citation = asCitation(part.data)
    if (citation) {
      citations.push(citation)
    }
  }
  return citations
}

export function statusesOf(message: CopilotMessage): PipelineStatus[] {
  const statuses: PipelineStatus[] = []
  for (const part of message.parts) {
    if (part.type !== "data-status" || !("data" in part)) {
      continue
    }
    const status = asStatus(part.data)
    if (status) {
      statuses.push(status)
    }
  }
  return statuses
}

export function lacksEvidence(text: string): boolean {
  const normalized = text.toLowerCase().replace(/\s+/g, " ")
  return (
    normalized.includes("not enough evidence") ||
    normalized.includes("does not contain enough evidence")
  )
}

export function citationLabel(citation: CitationData): string {
  const parts = [
    citation.companyName || citation.ticker,
    citation.filingType,
    formatFilingDate(citation.filingDate),
  ]
  if (citation.page) {
    parts.push(`p. ${citation.page}`)
  }
  if (citation.section) {
    parts.push(citation.section)
  }
  return parts.filter((part) => part.length > 0).join(" · ")
}

export function formatFilingDate(value: string): string {
  if (!value) {
    return ""
  }
  const date = new Date(`${value.slice(0, 10)}T00:00:00Z`)
  if (Number.isNaN(date.getTime())) {
    return value
  }
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
    timeZone: "UTC",
  }).format(date)
}

function toPart(part: StoredPart): CopilotMessage["parts"][number] | null {
  if (part.type === "text" && typeof part.text === "string") {
    return { type: "text", text: part.text }
  }
  if (part.type === "data-citation") {
    const citation = asCitation(part.data)
    if (!citation) {
      return null
    }
    return { type: "data-citation", id: part.id, data: citation }
  }
  if (part.type === "data-status") {
    const status = asStatus(part.data)
    if (!status) {
      return null
    }
    return { type: "data-status", id: part.id, data: status }
  }
  return null
}

function asCitation(value: unknown): CitationData | null {
  if (!isRecord(value) || typeof value.excerpt !== "string") {
    return null
  }
  const chunkId = text(value.chunkId)
  if (!chunkId) {
    return null
  }
  const fiscalYear = value.fiscalYear
  return {
    chunkId,
    ticker: text(value.ticker),
    companyName: text(value.companyName),
    filingType: text(value.filingType),
    filingDate: text(value.filingDate),
    fiscalYear: typeof fiscalYear === "number" ? fiscalYear : null,
    accessionNumber: text(value.accessionNumber),
    page: text(value.page) || null,
    section: text(value.section) || null,
    excerpt: value.excerpt,
    content: typeof value.content === "string" ? value.content : undefined,
  }
}

function asStatus(value: unknown): PipelineStatus | null {
  if (!isRecord(value) || typeof value.phase !== "string" || typeof value.detail !== "string") {
    return null
  }
  return { phase: value.phase, detail: value.detail }
}

function text(value: unknown): string {
  return typeof value === "string" ? value : ""
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null
}
