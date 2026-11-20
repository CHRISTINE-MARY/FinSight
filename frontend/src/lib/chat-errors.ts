import { ApiError } from "@/lib/http"

export function describeRequestError(error: unknown, fallback: string): string {
  if (error instanceof ApiError) {
    if (error.status === 401) {
      return "Your session expired. Sign in again."
    }
    if (error.isNetworkError) {
      return "Network or CORS error. The API did not respond."
    }
    return error.message || fallback
  }
  if (error instanceof Error) {
    return describeChatError(error.message)
  }
  return fallback
}

export function describeChatError(message: string): string {
  if (/failed to fetch|networkerror|network request failed|load failed|cors/i.test(message)) {
    return "Network or CORS error. The API did not respond."
  }
  if (/\b401\b|unauthorized|jwt expired|session expired/i.test(message)) {
    return "Your session expired. Sign in again."
  }
  return message
}
