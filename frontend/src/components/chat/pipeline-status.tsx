import type { PipelineStatus } from "@/lib/chat"

export function PipelineStatusList({
  statuses,
  live,
}: {
  statuses: PipelineStatus[]
  live: boolean
}) {
  if (!live && statuses.length === 0) {
    return null
  }

  const list = (
    <ol className="mt-2 space-y-1">
      {statuses.map((status, index) => (
        <li key={`${status.phase}-${index}`} className="text-xs leading-5">
          <span className="font-medium capitalize">{status.phase}</span>
          <span className="text-muted-foreground"> — {status.detail}</span>
        </li>
      ))}
      {live && statuses.length === 0 ? (
        <li className="text-xs text-muted-foreground">Starting the turn…</li>
      ) : null}
    </ol>
  )

  if (live) {
    return (
      <div className="mt-3" role="status" aria-label="Pipeline">
        <p className="text-xs font-medium text-muted-foreground">Pipeline</p>
        {list}
      </div>
    )
  }

  return (
    <details className="mt-3">
      <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
        Pipeline
      </summary>
      {list}
    </details>
  )
}
