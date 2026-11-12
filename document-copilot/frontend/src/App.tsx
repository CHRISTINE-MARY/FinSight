import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"

import { AuthProvider } from "@/lib/auth"
import { AuthPage } from "@/pages/auth-page"
import { ChatEmpty } from "@/pages/chat-empty"
import { ChatLayout } from "@/pages/chat-layout"
import { ChatThreadPage } from "@/pages/chat-thread-page"
import { RequireAuth } from "@/pages/require-auth"

export default function App() {
  return (
    <BrowserRouter>
      <AuthProvider>
        <Routes>
          <Route path="/signup" element={<AuthPage key="signup" mode="signup" />} />
          <Route path="/login" element={<AuthPage key="login" mode="login" />} />
          <Route path="/" element={<Navigate to="/chats" replace />} />
          <Route
            element={
              <RequireAuth>
                <ChatLayout />
              </RequireAuth>
            }
          >
            <Route path="/chats" element={<ChatEmpty />} />
            <Route path="/chats/:threadId" element={<ChatThreadPage />} />
          </Route>
          <Route path="*" element={<Navigate to="/chats" replace />} />
        </Routes>
      </AuthProvider>
    </BrowserRouter>
  )
}
