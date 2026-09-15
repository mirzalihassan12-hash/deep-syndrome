import { NextRequest, NextResponse } from "next/server";
import { getPatientsCollection } from "@/lib/mongodb";
import { getSessionFromRequest } from "@/lib/auth";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const session = getSessionFromRequest(req);
  if (!session) {
    return NextResponse.json({ ok: false, error: "You must be logged in." }, { status: 401 });
  }

  try {
    const patients = await getPatientsCollection();
    // Doctors see only their own patients; admins can see every patient.
    const filter = session.role === "admin" ? {} : { doctor_id: session.id };
    const list = await patients.find(filter).sort({ created_at: -1 }).toArray();
    return NextResponse.json({
      ok: true,
      patients: list.map((p) => ({
        id: p._id.toString(),
        name: p.name,
        date_of_birth: p.date_of_birth,
        notes: p.notes,
        doctor_id: p.doctor_id,
        created_at: p.created_at,
      })),
    });
  } catch (err) {
    console.error("list patients error:", err);
    return NextResponse.json({ ok: false, error: "Server error while loading patients." }, { status: 500 });
  }
}

export async function POST(req: NextRequest) {
  const session = getSessionFromRequest(req);
  if (!session) {
    return NextResponse.json({ ok: false, error: "You must be logged in." }, { status: 401 });
  }

  const body = await req.json().catch(() => null);
  const name = typeof body?.name === "string" ? body.name.trim() : "";
  const dateOfBirth = typeof body?.dateOfBirth === "string" ? body.dateOfBirth : "";
  const notes = typeof body?.notes === "string" ? body.notes.trim() : "";

  if (!name) {
    return NextResponse.json({ ok: false, error: "Patient name is required." }, { status: 400 });
  }

  try {
    const patients = await getPatientsCollection();
    const result = await patients.insertOne({
      name,
      date_of_birth: dateOfBirth || null,
      notes,
      doctor_id: session.id,
      created_at: new Date(),
    });
    return NextResponse.json({ ok: true, id: result.insertedId.toString() });
  } catch (err) {
    console.error("create patient error:", err);
    return NextResponse.json({ ok: false, error: "Server error while saving patient." }, { status: 500 });
  }
}
