"use client";

import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:7860";

type ModelVote = { prediction: string; confidence: number };

type PredictResponse = {
  prediction: string;
  confidence: number;
  probabilities: { control: number; down_syndrome: number };
  models_used: string[];
  individual: Record<string, ModelVote>;
  demo_mode: boolean;
  message?: string;
};

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
  const fileInputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const autoTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const loadingRef = useRef(false);
  const toastId = useRef(0);

  const toast = (msg: string, err = false) => {
    const id = ++toastId.current;
    setToasts((t) => [...t, { id, msg, err }]);
    setTimeout(() => setToasts((t) => t.filter((x) => x.id !== id)), 4000);
  };

  useEffect(() => {
    fetch(`${API_URL}/health`)
      .then((r) => r.json())
      .then((d) => {
        if (d.models_ready === 0) toast("⚠️ No model weights — Demo Mode active");
        else toast(`✅ ${d.models_ready}/3 models loaded and ready`);
      })
      .catch(() => toast("⚠️ Could not reach the backend API", true));
  }, []);

  const runPredict = async (imageFile: File) => {
    setLoading(true);
    loadingRef.current = true;
    try {
      const fd = new FormData();
      fd.append("file", imageFile);
      const res = await fetch(`${API_URL}/predict`, { method: "POST", body: fd });
      if (!res.ok) {
        const e = await res.json().catch(() => ({}));
        throw new Error(e.detail || "Server error");
      }
      const data: PredictResponse = await res.json();
      setResult(data);
      toast(
        data.prediction === "Down Syndrome"
          ? `⚠️ Down Syndrome — ${data.confidence.toFixed(1)}% confidence`
          : `✅ Control — ${data.confidence.toFixed(1)}% confidence`,
        data.prediction === "Down Syndrome"
      );
    } catch (e) {
      toast("Error: " + (e as Error).message, true);
    } finally {
      setLoading(false);
      loadingRef.current = false;
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
              {/* Verdict banner */}
              <div
                className={`mb-5 flex flex-wrap items-center justify-between gap-4 rounded-2xl border p-6 ${
                  isDS ? "border-[#ef4444]/50 bg-[#ef4444]/10" : "border-[#10b981]/50 bg-[#10b981]/10"
                }`}
              >
                <div className="flex items-center gap-5">
                  <div className="text-4xl">{isDS ? "⚠️" : "✅"}</div>
                  <div>
                    <div className="mb-1 text-xs uppercase tracking-widest text-[#94a3b8]">
                      Ensemble Verdict
                    </div>
                    <div className={`text-2xl font-extrabold ${isDS ? "text-[#ef4444]" : "text-[#10b981]"}`}>
                      {result.prediction}
                    </div>
                    {result.demo_mode && (
                      <span className="mt-2 inline-flex items-center gap-1 rounded-full border border-[#f59e0b] bg-[#f59e0b]/10 px-3 py-1 text-xs font-semibold text-[#f59e0b]">
                        ⚠️ Demo Mode — no .pth weights loaded
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-center">
                  <div className={`text-4xl font-black ${isDS ? "text-[#ef4444]" : "text-[#10b981]"}`}>
                    {result.confidence.toFixed(1)}%
                  </div>
                  <div className="text-xs uppercase tracking-wide text-[#94a3b8]">Confidence</div>
                </div>
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
          <a href={`${API_URL}/docs`} className="text-[#6366f1] hover:underline">
            API Docs
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
