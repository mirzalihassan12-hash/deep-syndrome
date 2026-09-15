import { NextRequest, NextResponse } from "next/server";
import { getConfirmedSamplesCollection } from "@/lib/mongodb";
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
