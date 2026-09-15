"use client";

import { useEffect, useState } from "react";
import Link from "next/link";

type Stats = {
  doctorCount: number;
  patientCount: number;
  sampleCount: number;
  dsConfirmed: number;
  controlConfirmed: number;
};

export default function AdminPage() {
  const [doctor, setDoctor] = useState<{ name: string; email: string; role: string } | null>(null);
  const [authChecked, setAuthChecked] = useState(false);
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => r.json())
      .then((d) => setDoctor(d.doctor))
      .finally(() => setAuthChecked(true));
  }, []);

  useEffect(() => {
    if (doctor?.role === "admin") {
      fetch("/api/admin/stats")
        .then((r) => r.json())
        .then((d) => (d.ok ? setStats(d.stats) : setError(d.error)))
        .catch(() => setError("Could not load stats."));
    }
  }, [doctor]);

  return (
    <div className="min-h-screen bg-[#0a0e1a] px-4 py-10 text-[#f1f5f9]">
      <div className="mx-auto max-w-3xl">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-extrabold">🛡️ Admin Dashboard</h1>
          <Link href="/" className="text-sm text-[#6366f1] hover:underline">
            ← Back to screening tool
          </Link>
        </div>

        {!authChecked ? (
          <p className="text-[#94a3b8]">Checking session…</p>
        ) : !doctor ? (
          <div className="rounded-2xl border border-[#2a3550] bg-[#111827] p-6 text-sm text-[#94a3b8]">
            You need to log in first —{" "}
            <Link href="/" className="text-[#6366f1] hover:underline">
              go to the screening tool
            </Link>
            .
          </div>
        ) : doctor.role !== "admin" ? (
          <div className="rounded-2xl border border-[#f59e0b] bg-[#f59e0b]/10 p-6 text-sm text-[#f59e0b]">
            This account doesn&apos;t have admin access. Admin status is granted manually, not through
            self-registration — see the project README for how to promote an account.
          </div>
        ) : error ? (
          <div className="rounded-2xl border border-[#ef4444] bg-[#ef4444]/10 p-6 text-sm text-[#ef4444]">
            {error}
          </div>
        ) : !stats ? (
          <p className="text-[#94a3b8]">Loading stats…</p>
        ) : (
          <div className="grid grid-cols-2 gap-4 sm:grid-cols-3">
            {[
              { label: "Doctors", value: stats.doctorCount, icon: "🩺" },
              { label: "Patients", value: stats.patientCount, icon: "🧑‍⚕️" },
              { label: "Confirmed Samples", value: stats.sampleCount, icon: "📋" },
              { label: "Confirmed Down Syndrome", value: stats.dsConfirmed, icon: "⚠️" },
              { label: "Confirmed Control", value: stats.controlConfirmed, icon: "✅" },
            ].map((s) => (
              <div key={s.label} className="rounded-2xl border border-[#2a3550] bg-[#111827] p-5 text-center">
                <div className="mb-2 text-2xl">{s.icon}</div>
                <div className="text-3xl font-black">{s.value}</div>
                <div className="mt-1 text-xs uppercase tracking-wide text-[#94a3b8]">{s.label}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
