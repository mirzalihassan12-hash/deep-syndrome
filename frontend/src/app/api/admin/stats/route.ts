import { NextRequest, NextResponse } from "next/server";
import { getDoctorsCollection, getPatientsCollection, getConfirmedSamplesCollection } from "@/lib/mongodb";
import { getSessionFromRequest } from "@/lib/auth";

export const runtime = "nodejs";

export async function GET(req: NextRequest) {
  const session = getSessionFromRequest(req);
  if (!session) {
    return NextResponse.json({ ok: false, error: "You must be logged in." }, { status: 401 });
  }
  if (session.role !== "admin") {
    return NextResponse.json({ ok: false, error: "Admin access required." }, { status: 403 });
  }

  try {
    const doctors = await getDoctorsCollection();
    const patients = await getPatientsCollection();
    const samples = await getConfirmedSamplesCollection();

    const [doctorCount, patientCount, sampleCount, dsConfirmed, controlConfirmed] = await Promise.all([
      doctors.countDocuments({}),
      patients.countDocuments({}),
      samples.countDocuments({}),
      samples.countDocuments({ doctor_label: "Down Syndrome" }),
      samples.countDocuments({ doctor_label: "Control" }),
    ]);

    return NextResponse.json({
      ok: true,
      stats: { doctorCount, patientCount, sampleCount, dsConfirmed, controlConfirmed },
    });
  } catch (err) {
    console.error("admin stats error:", err);
    return NextResponse.json({ ok: false, error: "Server error while loading stats." }, { status: 500 });
  }
}
