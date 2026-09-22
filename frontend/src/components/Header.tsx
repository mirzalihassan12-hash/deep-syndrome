"use client";

import Link from "next/link";
import { useAuth } from "@/lib/useAuth";

export default function Header() {
  const { doctor, authChecked, logout } = useAuth();

  return (
    <header className="sticky top-0 z-50 flex flex-wrap items-center justify-between gap-3 border-b border-[#2a3550] bg-[#111827]/75 px-6 py-4 backdrop-blur-md sm:px-10">
      <Link href="/" className="flex items-center gap-3">
        <div className="flex h-10 w-10 items-center justify-center rounded-[10px] bg-gradient-to-br from-[#6366f1] to-[#8b5cf6] text-xl shadow-[0_0_22px_rgba(99,102,241,.28)]">
          🧬
        </div>
        <div>
          <div className="bg-gradient-to-r from-[#6366f1] to-[#06b6d4] bg-clip-text text-xl font-extrabold text-transparent">
            DeepSyndrome
          </div>
          <div className="text-[.68rem] uppercase tracking-widest text-[#94a3b8]">
            AI Diagnostic System
          </div>
        </div>
      </Link>

      <div className="hidden gap-2 md:flex">
        <span className="rounded-full border border-[#6366f1] bg-[#6366f1]/10 px-3 py-1 text-xs font-semibold text-[#6366f1]">
          ResNet-50
        </span>
        <span className="rounded-full border border-[#8b5cf6] bg-[#8b5cf6]/10 px-3 py-1 text-xs font-semibold text-[#8b5cf6]">
          EfficientNet-B3
        </span>
        <span className="rounded-full border border-[#06b6d4] bg-[#06b6d4]/10 px-3 py-1 text-xs font-semibold text-[#06b6d4]">
          ViT-S/16
        </span>
      </div>

      <div className="flex items-center gap-4 text-sm">
        {!authChecked ? null : doctor ? (
          <>
            <Link href="/patients" className="text-[#94a3b8] hover:text-[#6366f1]">
              Patients
            </Link>
            {doctor.role === "admin" && (
              <Link href="/admin" className="text-[#94a3b8] hover:text-[#6366f1]">
                Admin
              </Link>
            )}
            <span className="hidden text-[#94a3b8] sm:inline">
              Dr. <span className="font-semibold text-[#f1f5f9]">{doctor.name}</span>
            </span>
            <button onClick={logout} className="text-[#94a3b8] hover:text-[#ef4444]">
              Log out
            </button>
          </>
        ) : (
          <>
            <Link href="/login" className="text-[#94a3b8] hover:text-[#f1f5f9]">
              Log In
            </Link>
            <Link
              href="/register"
              className="rounded-lg bg-gradient-to-br from-[#6366f1] to-[#8b5cf6] px-4 py-1.5 font-semibold text-white shadow-[0_4px_14px_rgba(99,102,241,.28)]"
            >
              Sign Up
            </Link>
          </>
        )}
      </div>
    </header>
  );
}
