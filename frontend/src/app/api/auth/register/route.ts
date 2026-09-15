import { NextRequest, NextResponse } from "next/server";
import bcrypt from "bcryptjs";
import { getDoctorsCollection } from "@/lib/mongodb";
import { signSession, sessionCookieOptions, SESSION_COOKIE } from "@/lib/auth";

export const runtime = "nodejs";

const EMAIL_RE = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

export async function POST(req: NextRequest) {
  const body = await req.json().catch(() => null);
  const name = typeof body?.name === "string" ? body.name.trim() : "";
  const emailRaw = typeof body?.email === "string" ? body.email.trim() : "";
  const password = typeof body?.password === "string" ? body.password : "";

  if (!name) {
    return NextResponse.json({ ok: false, error: "Name is required." }, { status: 400 });
  }
  if (!EMAIL_RE.test(emailRaw)) {
    return NextResponse.json({ ok: false, error: "Enter a valid email address." }, { status: 400 });
  }
  if (password.length < 8) {
    return NextResponse.json(
      { ok: false, error: "Password must be at least 8 characters." },
      { status: 400 }
    );
  }

  const email = emailRaw.toLowerCase();

  try {
    const doctors = await getDoctorsCollection();
    const existing = await doctors.findOne({ email });
    if (existing) {
      return NextResponse.json(
        { ok: false, error: "An account with this email already exists." },
        { status: 409 }
      );
    }

    const password_hash = await bcrypt.hash(password, 10);
    // Public registration is always "doctor" - admin status is granted
    // manually (see docs), never self-assignable through this form.
    const result = await doctors.insertOne({
      name,
      email,
      password_hash,
      role: "doctor",
      created_at: new Date(),
    });

    const token = signSession({ id: result.insertedId.toString(), email, name, role: "doctor" });
    const res = NextResponse.json({ ok: true, doctor: { name, email, role: "doctor" } });
    res.cookies.set(SESSION_COOKIE, token, sessionCookieOptions);
    return res;
  } catch (err) {
    console.error("register error:", err);
    return NextResponse.json({ ok: false, error: "Server error while registering." }, { status: 500 });
  }
}
