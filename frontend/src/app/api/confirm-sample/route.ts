import { NextRequest, NextResponse } from "next/server";
import { ObjectId } from "mongodb";
import { getConfirmedSamplesCollection, getPatientsCollection } from "@/lib/mongodb";
import { getSessionFromRequest } from "@/lib/auth";

export const runtime = "nodejs";

// Vercel serverless functions cap request bodies around 4.5MB regardless of
// framework — stay comfortably under that even though the main upload flow
// allows up to 10MB.
const MAX_IMAGE_BYTES = 4 * 1024 * 1024;
const VALID_LABELS = ["Down Syndrome", "Control"];

export async function POST(req: NextRequest) {
  const session = getSessionFromRequest(req);
  if (!session) {
    return NextResponse.json({ ok: false, error: "You must be logged in to confirm a sample." }, { status: 401 });
  }

  try {
    const form = await req.formData();

    const image = form.get("image");
    const doctorLabel = form.get("doctorLabel");
    if (!(image instanceof Blob) || typeof doctorLabel !== "string" || !VALID_LABELS.includes(doctorLabel)) {
      return NextResponse.json({ ok: false, error: "Missing image or invalid label." }, { status: 400 });
    }
    if (image.size > MAX_IMAGE_BYTES) {
      return NextResponse.json({ ok: false, error: "Image too large to store (max 4 MB)." }, { status: 413 });
    }

    // Optional link to a patient record - validated (not just trusted) so a
    // doctor can't attach a sample to another doctor's patient.
    const patientIdRaw = form.get("patientId");
    let patientId: string | null = null;
    if (typeof patientIdRaw === "string" && patientIdRaw) {
      if (!ObjectId.isValid(patientIdRaw)) {
        return NextResponse.json({ ok: false, error: "Invalid patient id." }, { status: 400 });
      }
      const patients = await getPatientsCollection();
      const patient = await patients.findOne({ _id: new ObjectId(patientIdRaw) });
      if (!patient || (session.role !== "admin" && patient.doctor_id !== session.id)) {
        return NextResponse.json({ ok: false, error: "Patient not found or not yours." }, { status: 403 });
      }
      patientId = patientIdRaw;
    }

    const arrayBuffer = await image.arrayBuffer();
    const doc = {
      image_base64: Buffer.from(arrayBuffer).toString("base64"),
      content_type: image.type || "image/jpeg",
      doctor_label: doctorLabel,
      model_prediction: String(form.get("modelPrediction") ?? ""),
      model_confidence: Number(form.get("confidence")) || null,
      face_detected: form.get("faceDetected") === "true",
      doctor_id: session.id,
      doctor_email: session.email,
      patient_id: patientId,
      created_at: new Date(),
      source: "doctor-mode-v1",
    };

    const collection = await getConfirmedSamplesCollection();
    const result = await collection.insertOne(doc);
    return NextResponse.json({ ok: true, id: result.insertedId.toString() });
  } catch (err) {
    console.error("confirm-sample error:", err);
    return NextResponse.json({ ok: false, error: "Server error while saving sample." }, { status: 500 });
  }
}
