import { useEffect, useState, type ReactNode } from "react"
import type { Session } from "@supabase/supabase-js"

import { ensureAppUser } from "@/lib/account"
import { AuthContext } from "@/lib/auth-context"
import { supabase } from "@/lib/supabase"

export function AuthProvider({ children }: { children: ReactNode }) {
  const [session, setSession] = useState<Session | null>(null)
  const [loading, setLoading] = useState(true)
  const [accountError, setAccountError] = useState<string | null>(null)

  useEffect(() => {
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, next) => {
      setSession(next)
      setLoading(false)
    })
    return () => subscription.unsubscribe()
  }, [])

  useEffect(() => {
    const email = session?.user.email
    if (!session || !email) {
      return
    }
    let active = true
    ensureAppUser(session.user.id, email).catch((error: unknown) => {
      if (!active) {
        return
      }
      setAccountError(
        error instanceof Error ? error.message : "Could not save your account",
      )
    })
    return () => {
      active = false
    }
  }, [session])

  async function signOut() {
    setAccountError(null)
    const { error } = await supabase.auth.signOut()
    if (error) {
      setAccountError(error.message)
    }
  }

  return (
    <AuthContext.Provider value={{ session, loading, accountError, signOut }}>
      {children}
    </AuthContext.Provider>
  )
}
