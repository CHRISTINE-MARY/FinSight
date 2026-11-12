import { supabase } from "@/lib/supabase"

export async function ensureAppUser(userId: string, email: string): Promise<void> {
  const existing = await supabase
    .from("users")
    .select("id")
    .eq("id", userId)
    .maybeSingle()
  if (existing.error) {
    throw new Error(existing.error.message)
  }
  if (existing.data) {
    return
  }

  const inserted = await supabase.from("users").insert({ id: userId, email })
  if (inserted.error && inserted.error.code !== "23505") {
    throw new Error(inserted.error.message)
  }
}
