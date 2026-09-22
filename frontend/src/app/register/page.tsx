"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import Header from "@/components/Header";
import { useAuth } from "@/lib/useAuth";

export default function RegisterPage() {
  const router = useRouter();
  const { refresh } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const submit = async () => {
    if (!name || !email || !password) return;
    setSubmitting(true);
    setError(null);
    try {
      const res = await fetch("/api/auth/register", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, email, password }),
      });
      const data = await res.json();
      if (data.ok) {
        await refresh();
        router.push("/");
      } else {
        setError(data.error || "Registration failed.");
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
        <h1 className="mb-1 text-2xl font-extrabold">🩺 Doctor Sign Up</h1>
        <p className="mb-6 text-sm text-[#94a3b8]">
          Register as a clinician to confirm predictions, manage patient records, and help improve the model.
        </p>

        <div className="rounded-2xl border border-[#2a3550] bg-[#111827] p-6">
          {error && (
            <div className="mb-4 rounded-lg border border-[#ef4444] bg-[#ef4444]/10 px-3 py-2 text-sm text-[#ef4444]">
              {error}
            </div>
          )}
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[#94a3b8]">Full name</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="mb-4 w-full rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
          />
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[#94a3b8]">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="mb-4 w-full rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
          />
          <label className="mb-1 block text-xs font-semibold uppercase tracking-wide text-[#94a3b8]">
            Password <span className="normal-case text-[#94a3b8]">(min 8 characters)</span>
          </label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && submit()}
            className="mb-6 w-full rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
          />
          <button
            onClick={submit}
            disabled={submitting || !name || !email || !password}
            className="w-full rounded-lg bg-gradient-to-br from-[#6366f1] to-[#06b6d4] py-3 font-bold shadow-[0_4px_20px_rgba(99,102,241,.35)] disabled:cursor-not-allowed disabled:opacity-40"
          >
            {submitting ? "Creating account…" : "Create Account"}
          </button>
        </div>

        <p className="mt-4 text-center text-sm text-[#94a3b8]">
          Already have an account?{" "}
          <Link href="/login" className="text-[#6366f1] hover:underline">
            Log in
          </Link>
        </p>
      </div>
    </div>
  );
}
