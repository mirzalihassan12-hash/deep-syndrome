"use client";

import { useCallback, useEffect, useState } from "react";

export type Doctor = { name: string; email: string; role?: string };

/** Session-aware doctor auth state, shared by the header and every page.
 * Each caller does its own /api/auth/me check (cheap, cookie-based) rather
 * than needing a global provider. */
export function useAuth() {
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [authChecked, setAuthChecked] = useState(false);

  const refresh = useCallback(() => {
    return fetch("/api/auth/me")
      .then((r) => r.json())
      .then((d) => setDoctor(d.doctor))
      .catch(() => setDoctor(null))
      .finally(() => setAuthChecked(true));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const logout = useCallback(async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch {
      // ignore network errors on logout - clear local state regardless
    }
    setDoctor(null);
  }, []);

  return { doctor, authChecked, logout, refresh };
}
