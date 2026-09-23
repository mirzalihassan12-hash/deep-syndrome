"use client";

import { createContext, useCallback, useContext, useEffect, useState } from "react";

export type Doctor = { name: string; email: string; role?: string };

type AuthState = {
  doctor: Doctor | null;
  authChecked: boolean;
  logout: () => Promise<void>;
  refresh: () => Promise<void>;
};

const AuthContext = createContext<AuthState | null>(null);

/** Single source of truth for doctor session state, shared by the header
 * and every page via context — so a login/register/logout on one page is
 * reflected everywhere immediately, without relying on a full page reload
 * or remount to pick up the change. */
export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [authChecked, setAuthChecked] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const r = await fetch("/api/auth/me");
      const d = await r.json();
      setDoctor(d.doctor);
    } catch {
      setDoctor(null);
    } finally {
      setAuthChecked(true);
    }
  }, []);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: checks the session cookie once on mount
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

  return <AuthContext.Provider value={{ doctor, authChecked, logout, refresh }}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within an AuthProvider");
  return ctx;
}
