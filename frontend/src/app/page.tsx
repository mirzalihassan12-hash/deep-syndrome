"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import { Client } from "@gradio/client";
import { runOfflineEnsemble, preloadOfflineModel } from "@/lib/offlineInference";

// Hugging Face Space reference, e.g. "username/space-name" or a full URL.
const HF_SPACE = process.env.NEXT_PUBLIC_HF_SPACE || "mirzalihassan12/syndrome-model";

type ModelVote = { prediction: string; confidence: number };

type PredictResponse = {
  prediction: string;
  confidence: number;
  probabilities: { control: number; down_syndrome: number };
  models_used: string[];
  individual: Record<string, ModelVote>;
  demo_mode: boolean;
  face_detected?: boolean;
  offline?: boolean;
  message?: string;
};

// Below this confidence, treat the result as "uncertain" rather than a
// clear verdict. Justified by our own held-out test: every misclassification
// we found was a low-confidence call, not a confident mistake - so a
// screening tool used in the field should say "uncertain, get it checked"
// instead of presenting a coin-flip as a confident answer either way.
const UNCERTAIN_THRESHOLD = 65;

type Toast = { id: number; msg: string; err: boolean };

const MODELS = [
  { name: "ResNet-50", key: "resnet", icon: "🏗️" },
  { name: "EfficientNet-B3", key: "eff", icon: "⚡" },
  { name: "ViT-S/16", key: "vit", icon: "👁️" },
] as const;

export default function Home() {
  const [mode, setMode] = useState<"upload" | "camera">("upload");
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [toasts, setToasts] = useState<Toast[]>([]);
  const [dragOver, setDragOver] = useState(false);
  const [cameraReady, setCameraReady] = useState(false);
  const [autoDetect, setAutoDetect] = useState(false);
  const [lastFile, setLastFile] = useState<File | null>(null);
  const [offlineReady, setOfflineReady] = useState(false);
  const [doctor, setDoctor] = useState<{ name: string; email: string; role?: string } | null>(null);
  const [patients, setPatients] = useState<{ id: string; name: string }[]>([]);
  const [selectedPatientId, setSelectedPatientId] = useState<string>("");
  const [authChecked, setAuthChecked] = useState(false);
  const [authMode, setAuthMode] = useState<"login" | "register">("login");
  const [authForm, setAuthForm] = useState({ name: "", email: "", password: "" });
  const [authSubmitting, setAuthSubmitting] = useState(false);
  const [savingLabel, setSavingLabel] = useState<"Down Syndrome" | "Control" | null>(null);
  const [savedSampleId, setSavedSampleId] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const autoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const loadingRef = useRef(false);
  const toastId = useRef(0);
  const clientRef = useRef<Client | null>(null);

  const toast = (msg: string, err = false) => {
    const id = ++toastId.current;
    setToasts((t) => [...t, { id, msg, err }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  };

  const getClient = async () => {
    if (!clientRef.current) {
      clientRef.current = await Client.connect(HF_SPACE);
    }
    return clientRef.current;
  };

  useEffect(() => {
    getClient()
      .then(() => toast("✅ Connected to model backend"))
      .catch(() => toast("⚠️ Could not reach the Hugging Face Space", true));
  }, []);

  useEffect(() => {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/sw.js").catch(() => {});
    }
    // Warm up the offline model in the background once the page is idle, so
    // it's cached for offline use without slowing down the initial page load.
    const idleId = setTimeout(() => {
      preloadOfflineModel().then((ok) => {
        if (ok) {
          setOfflineReady(true);
          toast("✅ Offline mode ready — all 3 models cached");
        }
      });
    }, 1500);

    const onOffline = () => toast("📴 Offline — predictions will run on-device", true);
    const onOnline = () => toast("✅ Back online — using the full 3-model ensemble again");
    window.addEventListener("offline", onOffline);
    window.addEventListener("online", onOnline);
    return () => {
      clearTimeout(idleId);
      window.removeEventListener("offline", onOffline);
      window.removeEventListener("online", onOnline);
    };
  }, []);

  useEffect(() => {
    fetch("/api/auth/me")
      .then((r) => r.json())
      .then((d) => setDoctor(d.doctor))
      .catch(() => setDoctor(null))
      .finally(() => setAuthChecked(true));
  }, []);

  useEffect(() => {
    if (!doctor) {
      // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: clears the patient list on logout
      setPatients([]);
      return;
    }
    fetch("/api/patients")
      .then((r) => r.json())
      .then((d) => setPatients(d.ok ? d.patients : []))
      .catch(() => setPatients([]));
  }, [doctor]);

  const runPredict = async (imageFile: File) => {
    setLastFile(imageFile);
    setLoading(true);
    loadingRef.current = true;
    try {
      let data: PredictResponse;
      if (!navigator.onLine) {
        toast("📴 Offline — running the 3 models on-device (no face-crop). First run takes a few seconds…");
        data = await runOfflineEnsemble(imageFile);
      } else {
        // navigator.onLine only reflects the OS network interface, not actual
        // reachability - it can report "online" even when the HF Space is
        // unreachable (rate-limited, cold, or a real outage). @gradio/client
        // also doesn't always reject cleanly when the network is down (it can
        // log internal errors and hang rather than throw) - so a plain
        // try/catch isn't enough. Race it against a timeout instead, and fall
        // back to the on-device model whichever way it fails to respond.
        const withTimeout = <T,>(p: Promise<T>, ms: number) =>
          new Promise<T>((resolve, reject) => {
            const t = setTimeout(() => reject(new Error("timed out")), ms);
            p.then((v) => { clearTimeout(t); resolve(v); }, (e) => { clearTimeout(t); reject(e); });
          });

        try {
          const client = await withTimeout(getClient(), 8000);
          const res = await withTimeout(client.predict("/run_prediction", [imageFile]), 20000);
          [, , data] = res.data as [unknown, unknown, PredictResponse];
        } catch {
          toast("⚠️ Online model unreachable — running the models on-device instead", true);
          data = await runOfflineEnsemble(imageFile);
        }
      }
      setResult(data);
      setSavedSampleId(null);
      if (data.confidence < UNCERTAIN_THRESHOLD) {
        toast(`❓ Uncertain (${data.confidence.toFixed(1)}%) — recommend professional evaluation`, false);
      } else {
        toast(
          data.prediction === "Down Syndrome"
            ? `⚠️ Down Syndrome — ${data.confidence.toFixed(1)}% confidence`
            : `✅ Control — ${data.confidence.toFixed(1)}% confidence`,
          data.prediction === "Down Syndrome"
        );
      }
    } catch (e) {
      toast("Error: " + (e as Error).message, true);
    } finally {
      setLoading(false);
      loadingRef.current = false;
    }
  };

  const submitAuth = async () => {
    if (!authForm.email || !authForm.password) return;
    if (authMode === "register" && !authForm.name) return;
    setAuthSubmitting(true);
    try {
      const endpoint = authMode === "register" ? "/api/auth/register" : "/api/auth/login";
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(authForm),
      });
      const data = await res.json();
      if (data.ok) {
        setDoctor(data.doctor);
        setAuthForm({ name: "", email: "", password: "" });
        toast(`🩺 ${authMode === "register" ? "Registered" : "Logged in"} as Dr. ${data.doctor.name}`);
      } else {
        toast(data.error || "Authentication failed.", true);
      }
    } catch (e) {
      toast("Error: " + (e as Error).message, true);
    } finally {
      setAuthSubmitting(false);
    }
  };

  const logout = async () => {
    try {
      await fetch("/api/auth/logout", { method: "POST" });
    } catch {
      // ignore network errors on logout - clear local state regardless
    }
    setDoctor(null);
  };

  const confirmSample = async (label: "Down Syndrome" | "Control") => {
    if (!lastFile || !result) return;
    setSavingLabel(label);
    try {
      const body = new FormData();
      body.append("image", lastFile, lastFile.name || "sample.jpg");
      body.append("doctorLabel", label);
      body.append("modelPrediction", result.prediction);
      body.append("confidence", String(result.confidence));
      body.append("faceDetected", String(result.face_detected ?? true));
      if (selectedPatientId) body.append("patientId", selectedPatientId);

      const res = await fetch("/api/confirm-sample", { method: "POST", body });
      const data = await res.json();
      if (data.ok) {
        setSavedSampleId(data.id);
        toast("✔️ Saved for future training");
      } else {
        if (res.status === 401) setDoctor(null); // stale/invalid session - re-show login
        toast(data.error || "Failed to save sample.", true);
      }
    } catch (e) {
      toast("Error saving sample: " + (e as Error).message, true);
    } finally {
      setSavingLabel(null);
    }
  };

  const stopCamera = () => {
    streamRef.current?.getTracks().forEach((t) => t.stop());
    streamRef.current = null;
    setCameraReady(false);
    setAutoDetect(false);
    if (autoTimerRef.current) clearInterval(autoTimerRef.current);
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraReady(true);
    } catch {
      toast("⚠️ Could not access camera — check browser permissions", true);
      setMode("upload");
    }
  };

  const captureFrame = () => {
    const video = videoRef.current;
    if (!video || video.readyState < 2) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0);
    canvas.toBlob((blob) => {
      if (blob) runPredict(new File([blob], "capture.jpg", { type: "image/jpeg" }));
    }, "image/jpeg", 0.9);
  };

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect -- intentional: mounts/tears down the camera stream when switching modes
    if (mode === "camera") startCamera();
    else stopCamera();
    return () => stopCamera();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode]);

  useEffect(() => {
    if (!autoDetect) {
      if (autoTimerRef.current) clearInterval(autoTimerRef.current);
      return;
    }
    autoTimerRef.current = setInterval(() => {
      if (!loadingRef.current) captureFrame();
    }, 3500);
    return () => {
      if (autoTimerRef.current) clearInterval(autoTimerRef.current);
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoDetect]);

  const pick = (f: File) => {
    if (!f.type.startsWith("image/")) return toast("Please select an image file.", true);
    if (f.size > 10 * 1024 * 1024) return toast("Image exceeds 10 MB limit.", true);
    setFile(f);
    setPreview(URL.createObjectURL(f));
    setResult(null);
  };

  const clear = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  };

  const analyse = () => file && runPredict(file);

  const isDS = result?.prediction === "Down Syndrome";
  const isUncertain = result ? result.confidence < UNCERTAIN_THRESHOLD : false;
  const noFaceDetected = result ? result.face_detected === false : false;
  const totalLoaded = result ? Object.keys(result.individual).length : 0;
  const dsVotes = result
    ? Object.values(result.individual).filter((v) => v.prediction === "Down Syndrome").length
    : 0;

  return (
    <div className="relative flex-1">
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{
          background:
            "radial-gradient(ellipse 80% 50% at 15% 0%, rgba(99,102,241,.13) 0%, transparent 65%), radial-gradient(ellipse 60% 70% at 85% 100%, rgba(6,182,212,.08) 0%, transparent 65%)",
        }}
      />
      <div className="relative z-10">
        {/* Header */}
        <header className="sticky top-0 z-50 flex flex-wrap items-center justify-between gap-3 border-b border-[#2a3550] bg-[#111827]/75 px-6 py-4 backdrop-blur-md sm:px-10">
          <div className="flex items-center gap-3">
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
          </div>
          <div className="hidden gap-2 sm:flex">
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
        </header>

        {/* Hero */}
        <section className="px-6 pb-10 pt-12 text-center">
          <div className="mx-auto mb-6 inline-flex items-center gap-2 rounded-full border border-[#2a3550] bg-[#1a2235]/60 px-4 py-1.5 text-xs text-[#94a3b8]">
            <span className="h-1.5 w-1.5 rounded-full bg-[#10b981] dot-blink" /> Ensemble · 3 Models · Live
            {offlineReady && <span className="text-[#10b981]">· Offline ready</span>}
          </div>
          <h1 className="mx-auto max-w-2xl text-4xl font-extrabold leading-tight sm:text-5xl">
            Detect{" "}
            <span className="bg-gradient-to-r from-[#6366f1] to-[#06b6d4] bg-clip-text text-transparent">
              Down Syndrome
            </span>
            <br />
            from Facial Imagery
          </h1>
          <p className="mx-auto mt-4 max-w-lg text-[#94a3b8]">
            Three independent deep learning architectures vote on each image. See every model&apos;s
            individual verdict alongside the ensemble consensus.
          </p>
        </section>

        <div className="mx-auto max-w-3xl px-4 pb-20">
          {/* Mode tabs */}
          <div className="mb-4 flex gap-2">
            <button
              onClick={() => setMode("upload")}
              className={`flex-1 rounded-xl border px-4 py-2.5 text-sm font-semibold transition ${
                mode === "upload"
                  ? "border-[#6366f1] bg-[#6366f1]/10 text-[#6366f1]"
                  : "border-[#2a3550] bg-[#111827] text-[#94a3b8]"
              }`}
            >
              📤 Upload Image
            </button>
            <button
              onClick={() => setMode("camera")}
              className={`flex-1 rounded-xl border px-4 py-2.5 text-sm font-semibold transition ${
                mode === "camera"
                  ? "border-[#6366f1] bg-[#6366f1]/10 text-[#6366f1]"
                  : "border-[#2a3550] bg-[#111827] text-[#94a3b8]"
              }`}
            >
              📷 Live Camera
            </button>
          </div>

          {mode === "upload" ? (
            /* Upload card */
            <div className="mb-6 rounded-2xl border border-[#2a3550] bg-[#111827] p-6">
              <div
                onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
                onDragLeave={() => setDragOver(false)}
                onDrop={(e) => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) pick(f); }}
                className={`relative cursor-pointer rounded-xl border-2 border-dashed bg-black/20 px-6 py-9 text-center transition ${
                  dragOver ? "border-[#6366f1] bg-[#6366f1]/10 shadow-[0_0_28px_rgba(99,102,241,.28)]" : "border-[#2a3550]"
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/*"
                  className="absolute inset-0 cursor-pointer opacity-0"
                  onChange={(e) => e.target.files?.[0] && pick(e.target.files[0])}
                />
                <span className="mb-3 block text-4xl">🖼️</span>
                <h3 className="mb-1 text-sm font-semibold">Drop facial image here</h3>
                <p className="text-xs text-[#94a3b8]">JPG, PNG, WEBP — max 10 MB</p>
                <button
                  type="button"
                  className="mt-4 inline-flex items-center gap-2 rounded-lg bg-gradient-to-br from-[#6366f1] to-[#8b5cf6] px-5 py-2 text-sm font-semibold shadow-[0_4px_14px_rgba(99,102,241,.28)]"
                >
                  Choose File
                </button>
              </div>

              {preview && (
                <div className="relative mt-4 overflow-hidden rounded-lg border border-[#2a3550]">
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img src={preview} alt="preview" className="block max-h-56 w-full object-cover" />
                  <button
                    onClick={clear}
                    className="absolute right-2 top-2 flex h-7 w-7 items-center justify-center rounded-full bg-black/70 text-white hover:bg-[#ef4444]"
                  >
                    ✕
                  </button>
                </div>
              )}

              <button
                onClick={analyse}
                disabled={!file || loading}
                className="mt-4 flex w-full items-center justify-center gap-2 rounded-lg bg-gradient-to-br from-[#6366f1] to-[#06b6d4] py-3 font-bold shadow-[0_4px_20px_rgba(99,102,241,.35)] disabled:cursor-not-allowed disabled:opacity-40"
              >
                {loading ? (
                  <>
                    <span className="spinner" /> Analysing…
                  </>
                ) : result ? (
                  "Analyse Again"
                ) : file ? (
                  "Analyse Image"
                ) : (
                  "Select an Image First"
                )}
              </button>
            </div>
          ) : (
            /* Camera card */
            <div className="mb-6 rounded-2xl border border-[#2a3550] bg-[#111827] p-6">
              <div className="relative overflow-hidden rounded-xl border border-[#2a3550] bg-black">
                <video
                  ref={videoRef}
                  playsInline
                  muted
                  className="block max-h-96 w-full -scale-x-100 object-contain"
                />
                {loading && (
                  <div className="absolute inset-0 flex items-center justify-center bg-black/50">
                    <span className="spinner" />
                  </div>
                )}
              </div>
              <p className="mt-2 text-center text-xs text-[#94a3b8]">
                Prediction takes a few seconds per frame (3 models on CPU) — this captures a snapshot,
                not smooth continuous video AI.
              </p>
              <div className="mt-4 flex gap-2">
                <button
                  onClick={captureFrame}
                  disabled={!cameraReady || loading}
                  className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-gradient-to-br from-[#6366f1] to-[#06b6d4] py-3 font-bold shadow-[0_4px_20px_rgba(99,102,241,.35)] disabled:cursor-not-allowed disabled:opacity-40"
                >
                  {loading ? (
                    <>
                      <span className="spinner" /> Analysing…
                    </>
                  ) : (
                    "📸 Capture & Analyse"
                  )}
                </button>
                <button
                  onClick={() => setAutoDetect((a) => !a)}
                  disabled={!cameraReady}
                  className={`rounded-lg border px-4 py-3 text-sm font-semibold disabled:cursor-not-allowed disabled:opacity-40 ${
                    autoDetect
                      ? "border-[#ef4444] bg-[#ef4444]/10 text-[#ef4444]"
                      : "border-[#2a3550] text-[#94a3b8]"
                  }`}
                >
                  {autoDetect ? "⏹ Stop auto (every 3.5s)" : "▶ Auto-detect every 3.5s"}
                </button>
              </div>
            </div>
          )}

          {/* Placeholder */}
          {!result && (
            <div className="mb-6 rounded-2xl border border-[#2a3550] bg-[#111827] px-6 py-14 text-center">
              <span className="mb-3 block text-5xl opacity-25">🔬</span>
              <p className="mx-auto max-w-64 text-sm text-[#94a3b8]">
                {mode === "camera"
                  ? "Capture a frame from your camera to see each model's vote here."
                  : "Upload a facial image and click Analyse. Each model's vote will appear here."}
              </p>
            </div>
          )}

          {/* Result */}
          {result && (
            <div>
              {/* No-face-detected warning */}
              {noFaceDetected && (
                <div className="mb-4 flex items-start gap-3 rounded-xl border border-[#f59e0b] bg-[#f59e0b]/10 p-4 text-sm">
                  <span className="text-lg">📷</span>
                  <div>
                    <div className="font-bold text-[#f59e0b]">No face detected in this image</div>
                    <div className="text-[#94a3b8]">
                      The result below is based on the full image rather than a cropped face, and is
                      less reliable. Try a clearer, front-facing photo.
                    </div>
                  </div>
                </div>
              )}

              {/* Verdict banner */}
              <div
                className={`mb-5 flex flex-wrap items-center justify-between gap-4 rounded-2xl border p-6 ${
                  isUncertain
                    ? "border-[#f59e0b]/50 bg-[#f59e0b]/10"
                    : isDS
                    ? "border-[#ef4444]/50 bg-[#ef4444]/10"
                    : "border-[#10b981]/50 bg-[#10b981]/10"
                }`}
              >
                <div className="flex items-center gap-5">
                  <div className="text-4xl">{isUncertain ? "❓" : isDS ? "⚠️" : "✅"}</div>
                  <div>
                    <div className="mb-1 text-xs uppercase tracking-widest text-[#94a3b8]">
                      {isUncertain ? "Result Uncertain" : "Ensemble Verdict"}
                    </div>
                    <div
                      className={`text-2xl font-extrabold ${
                        isUncertain ? "text-[#f59e0b]" : isDS ? "text-[#ef4444]" : "text-[#10b981]"
                      }`}
                    >
                      {isUncertain ? "Recommend Professional Evaluation" : result.prediction}
                    </div>
                    {isUncertain && (
                      <div className="mt-1 text-xs text-[#94a3b8]">
                        Models leaned toward &ldquo;{result.prediction}&rdquo;, but confidence was too
                        close to call reliably.
                      </div>
                    )}
                    {result.offline && (
                      <div className="mt-2 inline-flex items-center gap-1 rounded-full border border-[#6366f1] bg-[#6366f1]/10 px-3 py-1 text-xs font-semibold text-[#6366f1]">
                        📴 On-device result — {result.models_used.length}/3 models, no face-crop
                      </div>
                    )}
                    {result.demo_mode && (
                      <span className="mt-2 inline-flex items-center gap-1 rounded-full border border-[#f59e0b] bg-[#f59e0b]/10 px-3 py-1 text-xs font-semibold text-[#f59e0b]">
                        ⚠️ Demo Mode — no .pth weights loaded
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-center">
                  <div
                    className={`text-4xl font-black ${
                      isUncertain ? "text-[#f59e0b]" : isDS ? "text-[#ef4444]" : "text-[#10b981]"
                    }`}
                  >
                    {result.confidence.toFixed(1)}%
                  </div>
                  <div className="text-xs uppercase tracking-wide text-[#94a3b8]">Confidence</div>
                </div>
              </div>

              {/* Disclaimer - always shown with a result, not buried in the footer */}
              <div className="mb-5 flex items-start gap-3 rounded-xl border border-[#2a3550] bg-[#111827] p-4 text-sm">
                <span className="text-lg">ℹ️</span>
                <div className="text-[#94a3b8]">
                  <span className="font-bold text-[#f1f5f9]">This is a screening aid, not a medical
                  diagnosis.</span> Always consult a qualified healthcare professional for an accurate
                  assessment — regardless of what this tool shows.
                </div>
              </div>

              {/* Doctor Mode */}
              <div className="mb-5 rounded-2xl border border-[#2a3550] bg-[#111827] p-6">
                <div className="mb-3 flex items-center justify-between">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-[#94a3b8]">
                    🩺 Doctor Mode — Confirm Ground Truth
                  </div>
                  {doctor && (
                    <div className="flex items-center gap-3 text-xs">
                      <Link href="/patients" className="text-[#6366f1] hover:underline">
                        Patients
                      </Link>
                      {doctor.role === "admin" && (
                        <Link href="/admin" className="text-[#6366f1] hover:underline">
                          Admin
                        </Link>
                      )}
                      <button onClick={logout} className="text-[#94a3b8] hover:text-[#ef4444]">
                        Log out
                      </button>
                    </div>
                  )}
                </div>

                {!authChecked ? (
                  <p className="text-sm text-[#94a3b8]">Checking session…</p>
                ) : !doctor ? (
                  <div>
                    <div className="mb-3 flex gap-4 text-sm">
                      <button
                        onClick={() => setAuthMode("login")}
                        className={authMode === "login" ? "font-bold text-[#6366f1]" : "text-[#94a3b8]"}
                      >
                        Log in
                      </button>
                      <button
                        onClick={() => setAuthMode("register")}
                        className={authMode === "register" ? "font-bold text-[#6366f1]" : "text-[#94a3b8]"}
                      >
                        Register
                      </button>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      {authMode === "register" && (
                        <input
                          type="text"
                          value={authForm.name}
                          onChange={(e) => setAuthForm((f) => ({ ...f, name: e.target.value }))}
                          placeholder="Full name"
                          className="min-w-32 flex-1 rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
                        />
                      )}
                      <input
                        type="email"
                        value={authForm.email}
                        onChange={(e) => setAuthForm((f) => ({ ...f, email: e.target.value }))}
                        placeholder="Email"
                        className="min-w-40 flex-1 rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
                      />
                      <input
                        type="password"
                        value={authForm.password}
                        onChange={(e) => setAuthForm((f) => ({ ...f, password: e.target.value }))}
                        onKeyDown={(e) => e.key === "Enter" && submitAuth()}
                        placeholder="Password"
                        className="min-w-32 flex-1 rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
                      />
                      <button
                        onClick={submitAuth}
                        disabled={authSubmitting}
                        className="rounded-lg border border-[#6366f1] px-4 py-2 text-sm font-semibold text-[#6366f1] disabled:opacity-40"
                      >
                        {authSubmitting ? "…" : authMode === "register" ? "Create account" : "Log in"}
                      </button>
                    </div>
                    <p className="mt-3 text-xs text-[#94a3b8]">
                      For clinicians: confirming the real diagnosis helps retrain and improve the model.
                    </p>
                  </div>
                ) : (
                  <div>
                    <p className="mb-3 text-sm text-[#94a3b8]">
                      Logged in as <span className="font-semibold text-[#f1f5f9]">Dr. {doctor.name}</span>.
                      {savedSampleId ? "" : " What is the confirmed diagnosis for this image?"}
                    </p>
                    {savedSampleId ? (
                      <div className="text-sm font-semibold text-[#10b981]">✔️ Saved for future training</div>
                    ) : (
                      <div>
                        {patients.length > 0 && (
                          <select
                            value={selectedPatientId}
                            onChange={(e) => setSelectedPatientId(e.target.value)}
                            className="mb-3 w-full rounded-lg border border-[#2a3550] bg-black/20 px-3 py-2 text-sm"
                          >
                            <option value="">No patient record (don&apos;t link this sample)</option>
                            {patients.map((p) => (
                              <option key={p.id} value={p.id}>
                                {p.name}
                              </option>
                            ))}
                          </select>
                        )}
                        <div className="flex flex-wrap gap-2">
                        <button
                          onClick={() => confirmSample("Down Syndrome")}
                          disabled={!!savingLabel}
                          className="rounded-lg border border-[#ef4444] bg-[#ef4444]/10 px-4 py-2 text-sm font-semibold text-[#ef4444] disabled:opacity-40"
                        >
                          {savingLabel === "Down Syndrome" ? "Saving…" : "Confirm: Down Syndrome"}
                        </button>
                        <button
                          onClick={() => confirmSample("Control")}
                          disabled={!!savingLabel}
                          className="rounded-lg border border-[#10b981] bg-[#10b981]/10 px-4 py-2 text-sm font-semibold text-[#10b981] disabled:opacity-40"
                        >
                          {savingLabel === "Control" ? "Saving…" : "Confirm: Control"}
                        </button>
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>

              {/* Probability bars */}
              <div className="mb-5 rounded-2xl border border-[#2a3550] bg-[#111827] p-6">
                <div className="mb-4 text-xs font-bold uppercase tracking-widest text-[#94a3b8]">
                  📊 Class Probabilities
                </div>
                {[
                  { label: "✅ Control (No Down Syndrome)", pct: result.probabilities.control, color: "bg-[#10b981]" },
                  { label: "⚠️ Down Syndrome", pct: result.probabilities.down_syndrome, color: "bg-[#ef4444]" },
                ].map((row) => (
                  <div key={row.label} className="mb-4 last:mb-0">
                    <div className="mb-2 flex justify-between text-sm">
                      <span className="text-[#94a3b8]">{row.label}</span>
                      <span className="font-bold">{row.pct.toFixed(1)}%</span>
                    </div>
                    <div className="h-2.5 overflow-hidden rounded-full bg-[#2a3550]">
                      <div
                        className={`h-full rounded-full transition-all duration-1000 ${row.color}`}
                        style={{ width: `${row.pct}%` }}
                      />
                    </div>
                  </div>
                ))}
              </div>

              {/* Model votes */}
              <div className="overflow-hidden rounded-2xl border border-[#2a3550] bg-[#111827]">
                <div className="flex items-center justify-between border-b border-[#2a3550] px-6 py-4">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-[#94a3b8]">
                    🤖 Individual Model Votes
                  </div>
                  <span
                    className={`rounded-full border px-3 py-1 text-xs font-bold ${
                      totalLoaded === 0
                        ? "border-[#ef4444] bg-[#ef4444]/10 text-[#ef4444]"
                        : dsVotes === totalLoaded
                        ? "border-[#ef4444] bg-[#ef4444]/10 text-[#ef4444]"
                        : dsVotes === 0
                        ? "border-[#10b981] bg-[#10b981]/10 text-[#10b981]"
                        : "border-[#f59e0b] bg-[#f59e0b]/10 text-[#f59e0b]"
                    }`}
                  >
                    {totalLoaded === 0
                      ? "No models loaded"
                      : dsVotes === totalLoaded
                      ? "Unanimous — Down Syndrome"
                      : dsVotes === 0
                      ? "Unanimous — Control"
                      : `Split — ${dsVotes}/${totalLoaded} voted DS`}
                  </span>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-3">
                  {MODELS.map((m, i) => {
                    const info = result.individual[m.name];
                    const mDS = info?.prediction === "Down Syndrome";
                    return (
                      <div
                        key={m.key}
                        className={`p-6 text-center ${i < MODELS.length - 1 ? "sm:border-r border-[#2a3550]" : ""}`}
                      >
                        <span className="mb-2 block text-2xl">{m.icon}</span>
                        <div className="mb-3 text-xs font-bold uppercase tracking-wide text-[#94a3b8]">
                          {m.name}
                        </div>
                        {info ? (
                          <>
                            <div
                              className={`mb-3 inline-flex items-center rounded-full border-[1.5px] px-4 py-1.5 text-sm font-bold ${
                                mDS ? "border-[#ef4444] bg-[#ef4444]/10 text-[#ef4444]" : "border-[#10b981] bg-[#10b981]/10 text-[#10b981]"
                              }`}
                            >
                              {info.prediction}
                            </div>
                            <div className="mx-4 mb-2 h-1.5 overflow-hidden rounded-full bg-[#2a3550]">
                              <div
                                className={`h-full rounded-full transition-all duration-1000 ${mDS ? "bg-[#ef4444]" : "bg-[#10b981]"}`}
                                style={{ width: `${info.confidence}%` }}
                              />
                            </div>
                            <div className={`text-sm font-bold ${mDS ? "text-[#ef4444]" : "text-[#10b981]"}`}>
                              {info.confidence.toFixed(1)}% confident
                            </div>
                          </>
                        ) : (
                          <div className="py-4 text-sm text-[#94a3b8]">Not loaded</div>
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* How it works */}
        <section className="mx-auto max-w-3xl px-4 pb-16">
          <h2 className="mb-6 bg-gradient-to-r from-[#6366f1] to-[#06b6d4] bg-clip-text text-center text-xl font-extrabold text-transparent">
            How the Ensemble Works
          </h2>
          <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
            {[
              { n: 1, i: "📸", t: "Image Upload", d: "Facial image received and validated — type & size check." },
              { n: 2, i: "⚙️", t: "Preprocessing", d: "Resized to 224×224 and normalized with ImageNet mean/std." },
              { n: 3, i: "🧠", t: "Parallel Inference", d: "All three models run independently." },
              { n: 4, i: "📊", t: "Probability Average", d: "Softmax outputs averaged — highest probability wins." },
            ].map((s) => (
              <div key={s.n} className="rounded-xl border border-[#2a3550] bg-[#111827] p-5 text-center">
                <div className="mx-auto mb-3 flex h-8 w-8 items-center justify-center rounded-full bg-gradient-to-br from-[#6366f1] to-[#8b5cf6] text-xs font-extrabold">
                  {s.n}
                </div>
                <span className="mb-2 block text-xl">{s.i}</span>
                <div className="mb-1 text-sm font-bold">{s.t}</div>
                <div className="text-xs leading-relaxed text-[#94a3b8]">{s.d}</div>
              </div>
            ))}
          </div>
        </section>

        <footer className="border-t border-[#2a3550] px-4 py-7 text-center text-sm text-[#94a3b8]">
          DeepSyndrome v2.0 · Ensemble: ResNet-50 + EfficientNet-B3 + ViT-S/16 ·{" "}
          <a href={`https://huggingface.co/spaces/${HF_SPACE}`} className="text-[#6366f1] hover:underline">
            Model Space
          </a>
        </footer>
      </div>

      {/* Toasts */}
      <div className="fixed bottom-6 right-6 z-[999] flex flex-col gap-2">
        {toasts.map((t) => (
          <div
            key={t.id}
            className={`toast-in rounded-lg border bg-[#1a2235] px-4 py-3 text-sm shadow-lg ${
              t.err ? "border-[#ef4444]" : "border-[#2a3550]"
            }`}
          >
            {t.msg}
          </div>
        ))}
      </div>
    </div>
  );
}
