"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Header from "@/components/Header";
import { useAuth } from "@/lib/useAuth";

export default function LoginPage() {
  const router = useRouter();
  const { refresh } = useAuth();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!email || !password) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/login", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ email, password }),
      });
      const data = await res.json();
      if (data.ok) {
        await refresh();
        router.push("/");
      } else {
        setError(data.error || "Login failed.");
      }
    } catch (e) {
      setError("Error: " + (e as Error).message);
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-[#f1f5f9]">
      <Header />
      <div className="mx-auto flex max-w-md flex-col px-4 py-16">
        <h1 className="mb-1 text-2xl font-extrabold">🩺 Doctor Log In</h1>
        <p className="mb-6 text-sm text-[#94a3b8]">
          For clinicians: confirming the real diagnosis on a prediction helps retrain and improve the model.
        </p>

        <div className="rounded-2xl border border-[#2a3550] bg-[#111827] p-6">
          {error && (
            <div className="mb-4 rounded-lg border border-[#ef4444] bg-[#ef4444]/10 px-3 py-2 text-sm text-[#ef4444]">
              {error}
            </div>
          )}
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[#94a3b8]">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mb-4 w-full rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
          />
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[#94a3b8]">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            className="mb-6 w-full rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
          />
          <button
            onClick={submit}
            disabled={submitting || !email || !password}
            className="w-full rounded-lg bg-gradient-to-br from-[#6366f1] to-[#06b6d4] py-3 font-bold shadow-[0_4px_20px_rgba(99,102,241,.35)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? "Logging in…" : "Log In"}
          </button>
        </div>

        <p className="mt-4 text-center text-sm text-[#94a3b8]">
          Don&apos;t have an account?{" "}
          <Link href="/register" className="text-[#6366f1] hover:underline">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}
