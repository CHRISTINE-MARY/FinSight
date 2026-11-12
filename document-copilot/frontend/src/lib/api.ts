import { request } from "@/lib/http"

export const api = {
  get<T>(path: string): Promise<T> {
    return request<T>(path)
  },

  post<T>(path: string, body?: unknown): Promise<T> {
    return request<T>(path, {
      method: "POST",
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  },

  put<T>(path: string, body?: unknown): Promise<T> {
    return request<T>(path, {
      method: "PUT",
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  },

  patch<T>(path: string, body?: unknown): Promise<T> {
    return request<T>(path, {
      method: "PATCH",
      body: body === undefined ? undefined : JSON.stringify(body),
    })
  },

  delete<T>(path: string): Promise<T> {
    return request<T>(path, { method: "DELETE" })
  },
}
