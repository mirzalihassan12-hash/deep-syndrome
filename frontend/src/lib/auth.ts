import jwt from "jsonwebtoken";
import type { NextRequest } from "next/server";

export const SESSION_COOKIE = "doctor_session";
const SESSION_MAX_AGE_SECONDS = 60 * 60 * 24 * 7; // 7 days

export type SessionDoctor = { id: string; email: string; name: string };

function getSecret(): string {
  const secret = process.env.JWT_SECRET;
  if (!secret) throw new Error("Missing JWT_SECRET environment variable");
  return secret;
}

export function signSession(doctor: SessionDoctor): string {
  return jwt.sign(doctor, getSecret(), { expiresIn: SESSION_MAX_AGE_SECONDS });
}

export function verifySession(token: string): SessionDoctor | null {
  try {
    const decoded = jwt.verify(token, getSecret());
    if (
      typeof decoded === "object" &&
      decoded !== null &&
      "id" in decoded &&
      "email" in decoded &&
      "name" in decoded
    ) {
      return { id: String(decoded.id), email: String(decoded.email), name: String(decoded.name) };
    }
    return null;
  } catch {
    return null;
  }
}

export function getSessionFromRequest(req: NextRequest): SessionDoctor | null {
  const token = req.cookies.get(SESSION_COOKIE)?.value;
  if (!token) return null;
  return verifySession(token);
}

export const sessionCookieOptions = {
  httpOnly: true,
  sameSite: "lax" as const,
  secure: process.env.NODE_ENV === "production",
  path: "/",
  maxAge: SESSION_MAX_AGE_SECONDS,
};
