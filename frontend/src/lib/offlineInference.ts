"use client";

// Offline, in-browser inference using ONNX Runtime Web - runs the ViT-S/16
// model entirely on-device, no server call. Used when the browser has no
// network connection (the Hugging Face Space backend is unreachable).
//
// Deliberate simplification vs. the server path: no face-detection crop is
// applied here (the image is resized directly). Porting the Python
// OpenCV face-detector to JavaScript would risk behaving differently than
// what the model was actually trained against - documented as a known
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

const MODEL_URL = "/models/vit_s16.onnx";
const IMG_SIZE = 224;
const MEAN = [0.485, 0.456, 0.406];
const STD = [0.229, 0.224, 0.225];

let sessionPromise: Promise<OrtTypes.InferenceSession> | null = null;

function getSession(): Promise<OrtTypes.InferenceSession> {
  if (!sessionPromise) {
    sessionPromise = getOrt().then((ort) =>
      ort.InferenceSession.create(MODEL_URL, { executionProviders: ["wasm"] })
    );
  }
  return sessionPromise;
}

/** Preload the ONNX runtime chunk + model so they are cached for offline use.
 * Resolves true only when the model session is fully ready. */
export async function preloadOfflineModel(): Promise<boolean> {
  try {
    await getSession();
    return true;
  } catch {
    sessionPromise = null; // allow a retry on the next attempt
    return false;
  }
}

async function imageFileToTensor(file: File): Promise<OrtTypes.Tensor> {
  const bitmap = await createImageBitmap(file);
  const canvas = document.createElement("canvas");
  canvas.width = IMG_SIZE;
  canvas.height = IMG_SIZE;
  const ctx = canvas.getContext("2d");
  if (!ctx) throw new Error("Could not get canvas context");
  ctx.drawImage(bitmap, 0, 0, IMG_SIZE, IMG_SIZE);
  const { data } = ctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE); // RGBA, HWC

  // Convert to normalized float32 CHW, matching the Python preprocessing
  // exactly (pixel/255, then (x - MEAN) / STD per channel).
  const chw = new Float32Array(3 * IMG_SIZE * IMG_SIZE);
  const plane = IMG_SIZE * IMG_SIZE;
  for (let i = 0; i < plane; i++) {
    const r = data[i * 4] / 255;
    const g = data[i * 4 + 1] / 255;
    const b = data[i * 4 + 2] / 255;
    chw[i] = (r - MEAN[0]) / STD[0];
    chw[plane + i] = (g - MEAN[1]) / STD[1];
    chw[2 * plane + i] = (b - MEAN[2]) / STD[2];
  }

  const ort = await getOrt();
  return new ort.Tensor("float32", chw, [1, 3, IMG_SIZE, IMG_SIZE]);
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

export async function runOfflineViT(file: File): Promise<OfflineResult> {
  const session = await getSession();
  const tensor = await imageFileToTensor(file);
  const output = await session.run({ input: tensor });
  const logits = output.output.data as Float32Array;
  const [control, downSyndrome] = softmax(logits);

  const prediction = downSyndrome > control ? "Down Syndrome" : "Control";
  const confidence = Math.max(control, downSyndrome) * 100;

  return {
    prediction,
    confidence,
    probabilities: { control: control * 100, down_syndrome: downSyndrome * 100 },
    models_used: ["ViT-S/16"],
    individual: {
      "ViT-S/16": { prediction, confidence },
    },
    demo_mode: false,
    face_detected: undefined, // no face-crop step in offline mode
    offline: true,
  };
}
