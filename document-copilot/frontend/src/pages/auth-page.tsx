import { useState, type FormEvent } from "react"
import { Link, Navigate } from "react-router-dom"

import { Button } from "@/components/ui/button"
import {
  Card,
  CardContent,
  CardDescription,
  CardFooter,
  CardHeader,
} from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useAuth } from "@/lib/use-auth"
import { supabase } from "@/lib/supabase"

type AuthMode = "signup" | "login"

export function AuthPage({ mode }: { mode: AuthMode }) {
  const { session, loading } = useAuth()
  const [pending, setPending] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [confirmationEmail, setConfirmationEmail] = useState<string | null>(null)

  if (!loading && session) {
    return <Navigate to="/" replace />
  }

  async function onSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault()
    const form = new FormData(event.currentTarget)
    const email = String(form.get("email") ?? "").trim()
    const password = String(form.get("password") ?? "")
    const confirmPassword = String(form.get("confirmPassword") ?? "")

    if (mode === "signup" && password !== confirmPassword) {
      setError("Passwords do not match.")
      return
    }

    setError(null)
    setPending(true)
    if (mode === "signup") {
      const { data, error: signUpError } = await supabase.auth.signUp({
        email,
        password,
        options: { emailRedirectTo: `${window.location.origin}/` },
      })
      setPending(false)
      if (signUpError) {
        setError(signUpError.message)
        return
      }
      if (data.user && data.user.identities?.length === 0) {
        setError("An account with this email already exists. Sign in instead.")
        return
      }
      if (!data.session) {
        setConfirmationEmail(email)
      }
      return
    }

    const { error: signInError } = await supabase.auth.signInWithPassword({
      email,
      password,
    })
    setPending(false)
    if (signInError) {
      setError(signInError.message)
    }
  }

  const title = mode === "signup" ? "Create your account" : "Sign in"
  const description =
    mode === "signup"
      ? "Use your work email. Document Copilot is for Driftwood analysts."
      : "Sign in with the email and password you used to create your account."

  return (
    <main className="flex min-h-svh flex-col items-center px-4 py-8">
      <Card className="my-auto w-full max-w-sm">
        <CardHeader>
          <h1 className="font-heading text-lg leading-snug font-medium">{title}</h1>
          <CardDescription>{description}</CardDescription>
        </CardHeader>
        {confirmationEmail ? (
          <CardContent>
            <p className="text-sm text-foreground">
              Check {confirmationEmail} for a confirmation link. After you confirm,
              sign in here.
            </p>
          </CardContent>
        ) : (
          <form onSubmit={onSubmit}>
            <CardContent className="flex flex-col gap-4">
              <div className="flex flex-col gap-2">
                <Label htmlFor="email">Email</Label>
                <Input
                  id="email"
                  name="email"
                  type="email"
                  autoComplete="email"
                  required
                  className="h-9"
                />
              </div>
              <div className="flex flex-col gap-2">
                <Label htmlFor="password">Password</Label>
                <Input
                  id="password"
                  name="password"
                  type="password"
                  autoComplete={mode === "signup" ? "new-password" : "current-password"}
                  minLength={6}
                  required
                  className="h-9"
                />
              </div>
              {mode === "signup" ? (
                <div className="flex flex-col gap-2">
                  <Label htmlFor="confirmPassword">Confirm password</Label>
                  <Input
                    id="confirmPassword"
                    name="confirmPassword"
                    type="password"
                    autoComplete="new-password"
                    minLength={6}
                    required
                    className="h-9"
                  />
                </div>
              ) : null}
              {error ? (
                <p role="alert" className="text-sm text-destructive">
                  {error}
                </p>
              ) : null}
            </CardContent>
            <CardFooter className="mt-4 border-t-0 bg-transparent">
              <Button type="submit" size="lg" className="w-full" disabled={pending}>
                {pending ? "Please wait…" : mode === "signup" ? "Create account" : "Sign in"}
              </Button>
            </CardFooter>
          </form>
        )}
        <p className="px-(--card-spacing) text-sm text-muted-foreground">
          {mode === "signup" ? (
            <>
              Already have an account?{" "}
              <Link to="/login" className="text-foreground underline-offset-4 hover:underline">
                Sign in
              </Link>
            </>
          ) : (
            <>
              Need an account?{" "}
              <Link to="/signup" className="text-foreground underline-offset-4 hover:underline">
                Create one
              </Link>
            </>
          )}
        </p>
      </Card>
    </main>
  )
}
