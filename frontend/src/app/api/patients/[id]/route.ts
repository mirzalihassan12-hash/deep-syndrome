import { NextRequest, NextResponse } from "next/server";
import { ObjectId } from "mongodb";
import { getPatientsCollection, getConfirmedSamplesCollection } from "@/lib/mongodb";
import { getSessionFromRequest } from "@/lib/auth";

export const runtime = "nodejs";

export async function GET(req: NextRequest, { params }: { params: Promise<{ id: string }> }) {
  const session = getSessionFromRequest(req);
  if (!session) {
    return NextResponse.json({ ok: false, error: "You must be logged in." }, { status: 401 });
  }

  const { id } = await params;
  if (!ObjectId.isValid(id)) {
    return NextResponse.json({ ok: false, error: "Invalid patient id." }, { status: 400 });
  }

  try {
    const patients = await getPatientsCollection();
    const patient = await patients.findOne({ _id: new ObjectId(id) });
    if (!patient) {
      return NextResponse.json({ ok: false, error: "Patient not found." }, { status: 404 });
    }
    if (session.role !== "admin" && patient.doctor_id !== session.id) {
      return NextResponse.json({ ok: false, error: "Not authorized to view this patient." }, { status: 403 });
    }

    const samplesCollection = await getConfirmedSamplesCollection();
    const samples = await samplesCollection
      .find({ patient_id: id })
      .sort({ created_at: -1 })
      .project({ image_base64: 0 }) // history list doesn't need the full image payload
      .toArray();

    return NextResponse.json({
      ok: true,
      patient: {
        id: patient._id.toString(),
        name: patient.name,
        date_of_birth: patient.date_of_birth,
        notes: patient.notes,
        created_at: patient.created_at,
      },
      history: samples.map((s) => ({
        id: s._id.toString(),
        doctor_label: s.doctor_label,
        model_prediction: s.model_prediction,
        model_confidence: s.model_confidence,
        created_at: s.created_at,
      })),
    });
  } catch (err) {
    console.error("get patient error:", err);
    return NextResponse.json({ ok: false, error: "Server error while loading patient." }, { status: 500 });
  }
}
