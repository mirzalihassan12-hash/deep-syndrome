import { NextRequest, NextResponse } from "next/server";
import { getSessionFromRequest } from "@/lib/auth";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const session = getSessionFromRequest(req);
  if (!session) {
    return NextResponse.json({ doctor: null });
  }
  return NextResponse.json({
    doctor: { name: session.name, email: session.email, role: session.role },
  });
}
