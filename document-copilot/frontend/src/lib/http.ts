import { env } from "@/lib/env"
import { getAccessToken } from "@/lib/session"

const REQUEST_TIMEOUT_MS = 30_000

export class ApiError extends Error {
  readonly status: number
  readonly isNetworkError: boolean

  constructor(
    message: string,
    options: { status: number; isNetworkError?: boolean },
  ) {
    super(message)
    this.name = "ApiError"
    this.status = options.status
    this.isNetworkError = options.isNetworkError ?? false
  }
}

export async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers)
  if (init.body != null && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json")
  }
  const token = await getAccessToken()
  if (token) {
    headers.set("Authorization", `Bearer ${token}`)
  }

  let response: Response
  try {
    response = await fetch(`${env.apiBaseUrl}${path}`, {
      ...init,
      headers,
      signal: init.signal ?? AbortSignal.timeout(REQUEST_TIMEOUT_MS),
    })
  } catch (error) {
    throw new ApiError(
      error instanceof Error ? error.message : "Network request failed",
      { status: 0, isNetworkError: true },
    )
  }

  if (!response.ok) {
    let detail = response.statusText
    try {
      const body = (await response.json()) as { detail?: unknown }
      if (typeof body.detail === "string" && body.detail) {
        detail = body.detail
      }
    } catch {
      // The body was not JSON.
    }
    throw new ApiError(detail || "Request failed", { status: response.status })
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}
