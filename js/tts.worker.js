/**
 * OpenVoice TTS Worker
 * Runs Kokoro-82M ONNX model client-side in an isolated Web Worker.
 * Handles automatic text chunking and seamless WAV concatenation.
 */
import { KokoroTTS } from "https://esm.sh/kokoro-js";

let ttsInstance = null;
let currentDtype = null;

let currentDevice = null;

async function getOrInitTTS(dtype) {
  if (ttsInstance && currentDtype === dtype) {
    return ttsInstance;
  }

  // Detect whether WebGPU is available in this worker environment
  let canUseWebGPU = false;
  if (typeof navigator !== "undefined" && navigator.gpu) {
    try {
      const adapter = await navigator.gpu.requestAdapter();
      if (adapter) {
        canUseWebGPU = true;
      }
    } catch (e) {
      console.warn("Worker WebGPU requestAdapter failed:", e);
      canUseWebGPU = false;
    }
  }

  const progressCb = (p) => {
    if (p.status === "progress" && p.total > 0) {
      const pct = Math.round((p.loaded / p.total) * 100);
      self.postMessage({
        status: "progress",
        data: { percent: pct, file: p.file || "weights", message: `Downloading ${p.file || "weights"}: ${pct}%` }
      });
    } else if (p.status === "done") {
      self.postMessage({
        status: "progress",
        data: { percent: 100, file: "weights", message: "Weights cached locally! Ready to speak." }
      });
    }
  };

  // Attempt WebGPU hardware acceleration first
  if (canUseWebGPU) {
    try {
      self.postMessage({
        status: "progress",
        data: { percent: 0, file: "Connecting...", message: `Initializing ${dtype.toUpperCase()} on WebGPU (Hardware Accelerated 🚀)...` }
      });

      ttsInstance = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-ONNX", {
        dtype: dtype,
        device: "webgpu",
        progress_callback: progressCb
      });
      currentDevice = "webgpu";
      currentDtype = dtype;
      self.postMessage({ status: "device_ready", device: "webgpu" });
      return ttsInstance;
    } catch (gpuErr) {
      console.warn("WebGPU initialization failed, falling back to WASM CPU:", gpuErr);
      ttsInstance = null;
    }
  }

  // Fallback to WebAssembly CPU
  self.postMessage({
    status: "progress",
    data: { percent: 0, file: "Connecting...", message: `Initializing ${dtype.toUpperCase()} on WebAssembly Multi-Core CPU...` }
  });

  ttsInstance = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-ONNX", {
    dtype: dtype,
    device: "wasm",
    progress_callback: progressCb
  });
  currentDevice = "wasm";
  currentDtype = dtype;
  self.postMessage({ status: "device_ready", device: "wasm" });
  return ttsInstance;
}

/**
 * Splits text into natural sentence/clause chunks under maxChars to prevent model truncation.
 * Uses 450 characters (Kokoro-82M comfortably supports up to 510 tokens/phonemes, which is ~300+ words).
 * Keeping chunks around 400-450 chars drastically reduces forward-pass overhead by 50%+ and prevents choppy cadence.
 */
function chunkText(text, maxChars = 450) {
  const rawParagraphs = text.split(/\n+/).map(p => p.trim()).filter(Boolean);
  const chunks = [];

  for (const para of rawParagraphs) {
    const sentences = para.match(/[^.!?]+[.!?]+(\s|$)|[^.!?]+$/g) || [para];
    let current = "";

    for (let sentence of sentences) {
      sentence = sentence.trim();
      if (!sentence) continue;

      if ((current + " " + sentence).trim().length <= maxChars) {
        current = current ? current + " " + sentence : sentence;
      } else {
        if (current) chunks.push(current);
        if (sentence.length > maxChars) {
          const clauses = sentence.match(/[^,;:—]+[,;:—]+(\s|$)|[^,;:—]+$/g) || [sentence];
          let sub = "";
          for (let clause of clauses) {
            clause = clause.trim();
            if ((sub + " " + clause).trim().length <= maxChars) {
              sub = sub ? sub + " " + clause : clause;
            } else {
              if (sub) chunks.push(sub);
              sub = clause;
            }
          }
          if (sub) chunks.push(sub);
          current = "";
        } else {
          current = sentence;
        }
      }
    }
    if (current) chunks.push(current);
  }

  return chunks.filter(c => c.trim().length > 0);
}

/**
 * Creates standard 16-bit PCM WAV Blob from raw Float32Array samples.
 */
function createWavBlob(float32Array, sampleRate = 24000) {
  const buffer = new ArrayBuffer(44 + float32Array.length * 2);
  const view = new DataView(buffer);

  function writeString(offset, string) {
    for (let i = 0; i < string.length; i++) {
      view.setUint8(offset + i, string.charCodeAt(i));
    }
  }

  writeString(0, "RIFF");
  view.setUint32(4, 36 + float32Array.length * 2, true);
  writeString(8, "WAVE");

  writeString(12, "fmt ");
  view.setUint32(16, 16, true);
  view.setUint16(20, 1, true);
  view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true);
  view.setUint16(34, 16, true);

  writeString(36, "data");
  view.setUint32(40, float32Array.length * 2, true);

  let offset = 44;
  for (let i = 0; i < float32Array.length; i++, offset += 2) {
    let s = Math.max(-1, Math.min(1, float32Array[i]));
    view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
  }

  return new Blob([buffer], { type: "audio/wav" });
}

let currentJobId = null;
let cancelRequested = false;

self.onmessage = async (e) => {
  const { type, text, voice, dtype, speed, id } = e.data;

  if (type === "cancel") {
    cancelRequested = true;
    return;
  }

  if (type === "preload") {
    try {
      await getOrInitTTS(dtype || "q4");
      self.postMessage({ status: "ready", id });
    } catch (err) {
      self.postMessage({ status: "error", id, message: err.message || String(err) });
    }
    return;
  }

  if (type === "generate") {
    currentJobId = id;
    cancelRequested = false;

    try {
      const tts = await getOrInitTTS(dtype || "q8");
      const startTime = performance.now();
      const speedValue = speed ? Math.max(0.5, Math.min(2.0, parseFloat(speed))) : 1.0;

      const chunks = chunkText(text, 450);
      if (chunks.length === 0) {
        throw new Error("Please enter text to synthesize.");
      }

      const audioSegments = [];
      let sampleRate = 24000;

      for (let i = 0; i < chunks.length; i++) {
        // Check if user requested cancellation
        if (cancelRequested || currentJobId !== id) {
          if (audioSegments.length > 0) {
            const totalSamples = audioSegments.reduce((acc, seg) => acc + seg.length, 0);
            const combinedBuffer = new Float32Array(totalSamples);
            let offset = 0;
            for (const seg of audioSegments) {
              combinedBuffer.set(seg, offset);
              offset += seg.length;
            }
            const partialBlob = createWavBlob(combinedBuffer, sampleRate);
            const partialDuration = (totalSamples / sampleRate).toFixed(1);
            self.postMessage({
              status: "cancelled",
              id,
              partialBlob,
              partialDuration,
              chunksProcessed: i,
              totalChunks: chunks.length,
              message: `Synthesis stopped. Preserved ${partialDuration}s audio (${i}/${chunks.length} parts).`
            });
          } else {
            self.postMessage({
              status: "cancelled",
              id,
              message: "Synthesis cancelled."
            });
          }
          return;
        }

        const chunk = chunks[i];
        const progressPct = Math.round(((i + 1) / chunks.length) * 100);

        // Calculate dynamic estimated remaining time based on observed speed
        let estRemainingMsg = "";
        if (i > 0) {
          const avgChunkTime = (performance.now() - startTime) / i;
          const remainingChunks = chunks.length - i;
          const estSec = Math.max(1, Math.round((avgChunkTime * remainingChunks) / 1000));
          estRemainingMsg = ` (~${estSec}s remaining)`;
        }

        self.postMessage({
          status: "synthesizing",
          id,
          current: i + 1,
          total: chunks.length,
          percent: progressPct,
          message: `Synthesizing part ${i + 1} of ${chunks.length}${estRemainingMsg}...`
        });

        const res = await tts.generate(chunk, {
          voice: voice || "af_heart",
          speed: speedValue
        });

        if (res.sampling_rate) {
          sampleRate = res.sampling_rate;
        }

        if (res.audio) {
          audioSegments.push(res.audio);

          // Dispatch individual chunk immediately for real-time progressive streaming
          const chunkBlob = createWavBlob(res.audio, sampleRate);
          const chunkDuration = (res.audio.length / sampleRate).toFixed(1);
          self.postMessage({
            status: "chunk_ready",
            id,
            chunkIndex: i,
            totalChunks: chunks.length,
            chunkBlob,
            chunkDuration,
            percent: progressPct
          });

          // Insert 200ms natural pause between sentences
          if (i < chunks.length - 1) {
            const pauseSamples = new Float32Array(Math.round(sampleRate * 0.20));
            audioSegments.push(pauseSamples);
          }
        }
      }

      const totalSamples = audioSegments.reduce((acc, seg) => acc + seg.length, 0);
      const combinedBuffer = new Float32Array(totalSamples);
      let offset = 0;
      for (const seg of audioSegments) {
        combinedBuffer.set(seg, offset);
        offset += seg.length;
      }

      const totalDurationSec = (totalSamples / sampleRate).toFixed(1);
      const elapsed = ((performance.now() - startTime) / 1000).toFixed(2);
      const blob = createWavBlob(combinedBuffer, sampleRate);

      self.postMessage({
        status: "complete",
        id,
        blob,
        elapsed,
        totalDuration: totalDurationSec,
        chunksProcessed: chunks.length,
        message: `Generated ${totalDurationSec}s of audio in ${elapsed}s`
      });
    } catch (err) {
      console.error("Worker TTS error:", err);
      self.postMessage({
        status: "error",
        id,
        message: err.message || String(err)
      });
    }
  }
};
