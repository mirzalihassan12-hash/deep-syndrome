"use client";

import { useEffect, useState } from "react";
import Link from "next/link";
import Header from "@/components/Header";
import { useAuth } from "@/lib/useAuth";

type Patient = {
  id: string;
  name: string;
  date_of_birth: string | null;
  notes: string;
  created_at: string;
};

type HistoryEntry = {
  id: string;
  doctor_label: string;
  model_prediction: string;
  model_confidence: number | null;
  created_at: string;
};

export default function PatientsPage() {
  const { doctor, authChecked } = useAuth();
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(false);
  const [form, setForm] = useState({ name: "", dateOfBirth: "", notes: "" });
  const [submitting, setSubmitting] = useState(false);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [history, setHistory] = useState<HistoryEntry[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);

  const loadPatients = async () => {
    setLoading(true);
    try {
      const res = await fetch("/api/patients");
      const data = await res.json();
      if (data.ok) setPatients(data.patients);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: loads the doctor's patient list once auth resolves
    if (doctor) loadPatients();
  }, [doctor]);

  const addPatient = async () => {
    if (!form.name) return;
    setSubmitting(true);
    try {
      const res = await fetch("/api/patients", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(form),
      });
      const data = await res.json();
      if (data.ok) {
        setForm({ name: "", dateOfBirth: "", notes: "" });
        loadPatients();
      }
    } finally {
      setSubmitting(false);
    }
  };

  const toggleHistory = async (id: string) => {
    if (expandedId === id) {
      setExpandedId(null);
      return;
    }
    setExpandedId(id);
    setHistoryLoading(true);
    try {
      const res = await fetch(`/api/patients/${id}`);
      const data = await res.json();
      setHistory(data.ok ? data.history : []);
    } finally {
      setHistoryLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#0a0e1a] text-[#f1f5f9]">
      <Header />
      <div className="mx-auto max-w-3xl px-4 py-10">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-extrabold">🧑‍⚕️ Patient Records</h1>
          <Link href="/" className="text-sm text-[#6366f1] hover:underline">
            ← Back to screening tool
          </Link>
        </div>

        {!authChecked ? (
          <p className="text-[#94a3b8]">Checking session…</p>
        ) : !doctor ? (
          <div className="rounded-2xl border border-[#2a3550] bg-[#111827] p-6 text-sm text-[#94a3b8]">
            You need to log in as a doctor first —{" "}
            <Link href="/login" className="text-[#6366f1] hover:underline">
              log in
            </Link>{" "}
            or{" "}
            <Link href="/register" className="text-[#6366f1] hover:underline">
              sign up
            </Link>
            .
          </div>
        ) : (
          <>
            <div className="mb-6 rounded-2xl border border-[#2a3550] bg-[#111827] p-6">
              <div className="mb-3 text-xs font-bold uppercase tracking-widest text-[#94a3b8]">
                Add Patient
              </div>
              <div className="flex flex-wrap gap-2">
                <input
                  type="text"
                  value={form.name}
                  onChange={(e) => setForm((f) => ({ ...f, name: e.target.value }))}
                  placeholder="Full name"
                  className="min-w-40 flex-1 rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
                />
                <input
                  type="date"
                  value={form.dateOfBirth}
                  onChange={(e) => setForm((f) => ({ ...f, dateOfBirth: e.target.value }))}
                  className="rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
                />
                <input
                  type="text"
                  value={form.notes}
                  onChange={(e) => setForm((f) => ({ ...f, notes: e.target.value }))}
                  placeholder="Notes (optional)"
                  className="min-w-40 flex-1 rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
                />
                <button
                  onClick={addPatient}
                  disabled={submitting || !form.name}
                  className="rounded-lg border border-[#6366f1] px-4 py-2 text-sm font-semibold text-[#6366f1] disabled:opacity-40"
                >
                  {submitting ? "Saving…" : "Add"}
                </button>
              </div>
            </div>

            <div className="rounded-2xl border border-[#2a3550] bg-[#111827] p-6">
              <div className="mb-3 text-xs font-bold uppercase tracking-widest text-[#94a3b8]">
                {loading ? "Loading…" : `${patients.length} Patient${patients.length === 1 ? "" : "s"}`}
              </div>
              {patients.length === 0 && !loading && (
                <p className="text-sm text-[#94a3b8]">No patients yet — add one above.</p>
              )}
              <div className="flex flex-col gap-2">
                {patients.map((p) => (
                  <div key={p.id} className="rounded-lg border border-[#2a3550] p-3">
                    <button
                      onClick={() => toggleHistory(p.id)}
                      className="flex w-full items-center justify-between text-left"
                    >
                      <div>
                        <div className="font-semibold">{p.name}</div>
                        <div className="text-xs text-[#94a3b8]">
                          {p.date_of_birth ? `DOB: ${p.date_of_birth}` : "DOB not set"}
                          {p.notes ? ` · ${p.notes}` : ""}
                        </div>
                      </div>
                      <span className="text-xs text-[#6366f1]">
                        {expandedId === p.id ? "Hide history" : "View history"}
                      </span>
                    </button>
                    {expandedId === p.id && (
                      <div className="mt-3 border-t border-[#2a3550] pt-3">
                        {historyLoading ? (
                          <p className="text-xs text-[#94a3b8]">Loading history…</p>
                        ) : history.length === 0 ? (
                          <p className="text-xs text-[#94a3b8]">No confirmed screenings for this patient yet.</p>
                        ) : (
                          <div className="flex flex-col gap-2">
                            {history.map((h) => (
                              <div key={h.id} className="text-xs">
                                <span
                                  className={
                                    h.doctor_label === "Down Syndrome" ? "text-[#ef4444]" : "text-[#10b981]"
                                  }
                                >
                                  {h.doctor_label}
                                </span>{" "}
                                <span className="text-[#94a3b8]">
                                  (model said {h.model_prediction}
                                  {h.model_confidence != null ? ` at ${h.model_confidence.toFixed(1)}%` : ""}) —{" "}
                                  {new Date(h.created_at).toLocaleString()}
                                </span>
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
