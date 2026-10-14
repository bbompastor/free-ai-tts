/**
 * OpenVoice TTS Worker
 * Runs Kokoro-82M ONNX model client-side in an isolated Web Worker.
 * Handles automatic text chunking and seamless WAV concatenation.
 */
import { KokoroTTS } from "https://esm.sh/kokoro-js";

let ttsInstance = null;
let currentDtype = null;

async function getOrInitTTS(dtype) {
  if (ttsInstance && currentDtype === dtype) {
    return ttsInstance;
  }

  self.postMessage({
    status: "progress",
    data: { percent: 0, file: "Connecting...", message: `Initializing ${dtype.toUpperCase()} model runtime...` }
  });

  ttsInstance = await KokoroTTS.from_pretrained("onnx-community/Kokoro-82M-ONNX", {
    dtype: dtype,
    progress_callback: (p) => {
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
    }
  });

  currentDtype = dtype;
  return ttsInstance;
}

/**
 * Splits text into natural sentence/clause chunks under maxChars to prevent model truncation.
 */
function chunkText(text, maxChars = 240) {
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
          const clauses = sentence.match(/[^,;:]+[,;:]+(\s|$)|[^,;:]+$/g) || [sentence];
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

self.onmessage = async (e) => {
  const { type, text, voice, dtype, id } = e.data;

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
    try {
      const tts = await getOrInitTTS(dtype || "q8");
      const startTime = performance.now();

      const chunks = chunkText(text, 240);
      if (chunks.length === 0) {
        throw new Error("Please enter text to synthesize.");
      }

      const audioSegments = [];
      let sampleRate = 24000;

      for (let i = 0; i < chunks.length; i++) {
        const chunk = chunks[i];
        const progressPct = Math.round(((i + 1) / chunks.length) * 100);

        self.postMessage({
          status: "synthesizing",
          id,
          current: i + 1,
          total: chunks.length,
          percent: progressPct,
          message: `Synthesizing part ${i + 1} of ${chunks.length}...`
        });

        const res = await tts.generate(chunk, {
          voice: voice || "af_heart"
        });

        if (res.sampling_rate) {
          sampleRate = res.sampling_rate;
        }

        if (res.audio) {
          audioSegments.push(res.audio);
          // Insert 220ms natural pause between sentences
          if (i < chunks.length - 1) {
            const pauseSamples = new Float32Array(Math.round(sampleRate * 0.22));
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
