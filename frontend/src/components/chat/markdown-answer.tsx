import type { Components } from "react-markdown"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"

const components: Components = {
  h1: ({ children }) => <h2 className="font-semibold">{children}</h2>,
  h2: ({ children }) => <h2 className="font-semibold">{children}</h2>,
  h3: ({ children }) => <h3 className="font-semibold">{children}</h3>,
  p: ({ children }) => <p className="leading-relaxed">{children}</p>,
  ul: ({ children }) => <ul className="list-disc space-y-1 pl-5">{children}</ul>,
  ol: ({ children }) => <ol className="list-decimal space-y-1 pl-5">{children}</ol>,
  li: ({ children }) => <li className="leading-relaxed">{children}</li>,
  strong: ({ children }) => <strong className="font-semibold">{children}</strong>,
  a: ({ href, children }) => (
    <a
      href={href}
      target="_blank"
      rel="noreferrer"
      className="underline underline-offset-2"
    >
      {children}
    </a>
  ),
  blockquote: ({ children }) => (
    <blockquote className="border-l-2 border-border pl-3 text-muted-foreground">
      {children}
    </blockquote>
  ),
  table: ({ children }) => (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse border border-border text-left">
        {children}
      </table>
    </div>
  ),
  th: ({ children }) => (
    <th className="border border-border bg-muted px-2 py-1.5 font-medium">
      {children}
    </th>
  ),
  td: ({ children }) => (
    <td className="border border-border px-2 py-1.5 align-top">{children}</td>
  ),
  pre: ({ children }) => (
    <pre className="overflow-x-auto rounded-lg bg-muted p-3 font-mono text-xs">
      {children}
    </pre>
  ),
  code: ({ className, children }) =>
    className ? (
      <code className={className}>{children}</code>
    ) : (
      <code className="rounded bg-muted px-1 py-0.5 font-mono text-[0.85em]">
        {children}
      </code>
    ),
}

export function MarkdownAnswer({ text }: { text: string }) {
  if (!text) {
    return null
  }
  return (
    <div className="space-y-2">
      <ReactMarkdown
        skipHtml
        remarkPlugins={[remarkGfm]}
        disallowedElements={["img"]}
        components={components}
      >
        {text}
      </ReactMarkdown>
    </div>
  )
}
