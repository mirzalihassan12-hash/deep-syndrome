import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { getDoctorsCollection } from "@/lib/mongodb";
import { signSession, sessionCookieOptions, SESSION_COOKIE } from "@/lib/auth";

export const runtime = "nodejs";

const INVALID_MSG = "Invalid email or password.";

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  const emailRaw = typeof body?.email === "string" ? body.email.trim() : "";
  const password = typeof body?.password === "string" ? body.password : "";

  if (!emailRaw || !password) {
    return NextResponse.json({ ok: false, error: INVALID_MSG }, { status: 401 });
  }

  try {
    const doctors = await getDoctorsCollection();
    const doctor = await doctors.findOne({ email: emailRaw.toLowerCase() });
    if (!doctor) {
      return NextResponse.json({ ok: false, error: INVALID_MSG }, { status: 401 });
    }

    const valid = await bcrypt.compare(password, doctor.password_hash);
    if (!valid) {
      return NextResponse.json({ ok: false, error: INVALID_MSG }, { status: 401 });
    }

    const role = doctor.role === "admin" ? "admin" : "doctor";
    const token = signSession({
      id: doctor._id.toString(),
      email: doctor.email,
      name: doctor.name,
      role,
    });
    const res = NextResponse.json({
      ok: true,
      doctor: { name: doctor.name, email: doctor.email, role },
    });
    res.cookies.set(SESSION_COOKIE, token, sessionCookieOptions);
    return res;
  } catch (err) {
    console.error("login error:", err);
    return NextResponse.json({ ok: false, error: "Server error while logging in." }, { status: 500 });
  }
}
