import { useState, type FormEvent } from "react"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"

export function Composer({
  pending,
  onSend,
}: {
  pending: boolean
  onSend: (text: string) => Promise<void>
}) {
  const [text, setText] = useState("")

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const message = text.trim()
    if (!message || pending) {
      return
    }
    setText("")
    await onSend(message)
  }

  return (
    <form onSubmit={onSubmit} className="flex gap-2 border-t px-4 py-3">
      <Input
        name="message"
        value={text}
        onChange={(event) => setText(event.currentTarget.value)}
        placeholder="Ask about a filing"
        aria-label="Message"
        className="h-9"
        disabled={pending}
      />
      <Button type="submit" disabled={pending || text.trim() === ""}>
        Send
      </Button>
    </form>
  )
}
