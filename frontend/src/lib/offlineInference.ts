"use client";

// Offline, in-browser inference using ONNX Runtime Web - runs the same three
// models as the server ensemble (ResNet-50, EfficientNet-B3, ViT-S/16) entirely
// on-device and averages their softmax outputs exactly like the Python
// ensemble does. Used when the Hugging Face Space is unreachable.
//
// Deliberate simplification vs. the server path: no face-detection crop is
// applied here (the image is resized directly). Porting the Python
// OpenCV face-detector to JavaScript would risk behaving differently than
// what the models were actually trained against - documented as a known
// tradeoff of offline mode, not hidden.

// Loaded lazily (browser only): the WASM-only entry point resolves URLs at
// import time, which breaks Next's server-side prerender if imported at the
// top level. "onnxruntime-web/wasm" is also the plain CPU runtime - the
// default entry pulls in a WebGPU-capable build whose extra files we don't ship.
import type * as OrtTypes from "onnxruntime-web";
type Ort = typeof OrtTypes;

let ortPromise: Promise<Ort> | null = null;
function getOrt(): Promise<Ort> {
  if (!ortPromise) {
    ortPromise = import("onnxruntime-web/wasm").then((m) => {
      const ort = m as unknown as Ort;
      ort.env.wasm.wasmPaths = "/ort/";
      ort.env.wasm.numThreads = 1; // avoids requiring cross-origin-isolation headers
      return ort;
    });
  }
  return ortPromise;
}

// Names match the model cards in the UI and the server response.
export const OFFLINE_MODELS = [
  { name: "ResNet-50", url: "/models/resnet50.onnx" },
  { name: "EfficientNet-B3", url: "/models/efficientnet_b3.onnx" },
  { name: "ViT-S/16", url: "/models/vit_s16.onnx" },
] as const;

const IMG_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

// One lazily-created session per model. Sessions are created one at a time
// (not in parallel) to keep peak memory down on phones, and a model that
// fails to load is skipped rather than breaking the whole ensemble.
const sessionPromises = new Map<string, Promise<OrtTypes.InferenceSession>>();

function getSession(name: string, url: string): Promise<OrtTypes.InferenceSession> {
  let p = sessionPromises.get(name);
  if (!p) {
    p = getOrt().then((ort) => ort.InferenceSession.create(url, { executionProviders: ["wasm"] }));
    sessionPromises.set(name, p);
    p.catch(() => sessionPromises.delete(name)); // allow a retry next time
  }
  return p;
}

/** Make the offline files available without keeping models in memory:
 * loads the ONNX runtime chunk and waits until every model file is present
 * in the service-worker cache (fetching it if it isn't yet). Resolves true
 * only when everything needed for offline use is cached. */
export async function preloadOfflineModel(): Promise<boolean> {
  try {
    await getOrt();
    const cache = await caches.open("deepsyndrome-offline-v3");
    for (const m of OFFLINE_MODELS) {
      if (await cache.match(m.url)) continue;
      // Not cached yet: request it through the service worker (which caches
      // it) and wait until the entry actually appears.
      fetch(m.url).then((r) => r.body?.cancel()).catch(() => {});
      const deadline = Date.now() + 10 * 60 * 1000;
      while (!(await cache.match(m.url))) {
        if (Date.now() > deadline) return false;
        await new Promise((r) => setTimeout(r, 1000));
      }
    }
    return true;
  } catch {
    return false;
  }
}

async function imageFileToChw(file: File): Promise<Float32Array> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get canvas context");
  ctx.drawImage(bitmap, 0, 0, IMG_SIZE, IMG_SIZE);
  const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE); // RGBA, HWC

  // Normalised float32 CHW, matching the Python preprocessing exactly
  // (pixel/255, then (x - MEAN) / STD per channel).
  const chw = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    chw[i] = (data[i * 4] / 255 - MEAN[0]) / STD[0];
    chw[plane + i] = (data[i * 4 + 1] / 255 - MEAN[1]) / STD[1];
    chw[2 * plane + i] = (data[i * 4 + 2] / 255 - MEAN[2]) / STD[2];
  }
  return chw;
}

function softmax(logits: Float32Array): [number, number] {
  const max = Math.max(logits[0], logits[1]);
  const e0 = Math.exp(logits[0] - max);
  const e1 = Math.exp(logits[1] - max);
  const sum = e0 + e1;
  return [e0 / sum, e1 / sum];
}

export type OfflineResult = {
  prediction: "Down Syndrome" | "Control";
  confidence: number;
  probabilities: { control: number; down_syndrome: number };
  models_used: string[];
  individual: Record<string, { prediction: string; confidence: number }>;
  demo_mode: boolean;
  face_detected?: boolean;
  offline: true;
};

export async function runOfflineEnsemble(file: File): Promise<OfflineResult> {
  const ort = await getOrt();
  const chw = await imageFileToChw(file);

  const perModel: { name: string; probs: [number, number] }[] = [];
  for (const m of OFFLINE_MODELS) {
    try {
      const session = await getSession(m.name, m.url);
      const tensor = new ort.Tensor("float32", chw.slice(), [1, 3, IMG_SIZE, IMG_SIZE]);
      const out = await session.run({ input: tensor });
      perModel.push({ name: m.name, probs: softmax(out.output.data as Float32Array) });
    } catch {
      // Skip a model that failed to load/run (e.g. low device memory) and
      // ensemble whatever did work.
    }
  }
  if (perModel.length === 0) throw new Error("No on-device model could be loaded");

  // Average the per-model softmax outputs, like the Python ensemble.
  const control = perModel.reduce((s, p) => s + p.probs[0], 0) / perModel.length;
  const downSyndrome = perModel.reduce((s, p) => s + p.probs[1], 0) / perModel.length;
  const prediction = downSyndrome > control ? "Down Syndrome" : "Control";

  const individual: OfflineResult["individual"] = {};
  for (const p of perModel) {
    const isDs = p.probs[1] > p.probs[0];
    individual[p.name] = {
      prediction: isDs ? "Down Syndrome" : "Control",
      confidence: Math.max(p.probs[0], p.probs[1]) * 100,
    };
  }

  return {
    prediction,
    confidence: Math.max(control, downSyndrome) * 100,
    probabilities: { control: control * 100, down_syndrome: downSyndrome * 100 },
    models_used: perModel.map((p) => p.name),
    individual,
    demo_mode: false,
    face_detected: undefined, // no face-crop step in offline mode
    offline: true,
  };
}
