import type { ReactNode } from "react"
import { Navigate } from "react-router-dom"

import { useAuth } from "@/lib/use-auth"

export function RequireAuth({ children }: { children: ReactNode }) {
  const { session, loading } = useAuth()

  if (loading) {
    return (
      <main className="flex min-h-svh items-center justify-center px-4">
        <p className="text-sm text-muted-foreground">Checking your session…</p>
      </main>
    )
  }

  if (!session) {
    return <Navigate to="/login" replace />
  }

  return children
}
