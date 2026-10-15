---
title: How I Built a 100% Free, Private In-Browser AI Text-to-Speech Tool Using Kokoro-82M & WebGPU
published: true
tags: webdev, ai, javascript, opensource
canonical_url: https://puretts.com
---

Every time I needed to convert an article, chapter, or voiceover script into natural audio, I ran into the same frustrating wall:

* Services demanding $22/month subscriptions.
* Strict 500-character limits before a paywall pops up.
* Mandatory account signups and credit card forms.
* Privacy concerns over uploading sensitive text to third-party cloud servers.

I decided to see if modern browser standards (WebGPU, WebAssembly, and Web Workers) could handle state-of-the-art neural speech synthesis completely on the client side — with **zero server bills, zero paywalls, and zero limits**.

The result is **[PureTTS (puretts.com)](https://puretts.com)**.

Here is how I built it and the engineering hurdles I had to solve.

---

### The Tech Stack

* **Neural Model:** [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) (a lightweight 82M-parameter model delivering comparable quality to ElevenLabs).
* **Inference Engine:** `kokoro-js` compiled to ONNX WebAssembly / WebGPU.
* **Architecture:** Dedicated Web Worker (`tts.worker.js`) + Vanilla JavaScript frontend.
* **Hosting:** GitHub Pages with a custom domain on Porkbun.
* **Total Cloud Inference Cost:** **$0.00 / month**.

---

### Challenge 1: The Main Thread Freeze (Offloading to a Web Worker)

Running neural inference in JavaScript is computationally heavy. On initial prototypes, running the forward pass on the main browser thread caused the entire DOM to lock up:
* The text counter froze.
* Buttons became unresponsive.
* The browser tab warned: *"This page is slowing down your browser."*

To keep the UI at a buttery 60 FPS, all model loading, phoneme tokenization, and ONNX execution were isolated inside a dedicated **Web Worker**:

```javascript
// main.js - Communicating asynchronously with the worker
const ttsWorker = new Worker('tts.worker.js', { type: 'module' });

ttsWorker.postMessage({
  type: 'GENERATE_TTS',
  text: userText,
  voice: selectedVoice,
  speed: 1.0
});

ttsWorker.onmessage = (event) => {
  const { type, progress, wavBlob, error } = event.data;
  if (type === 'PROGRESS') {
    updateProgressBar(progress);
  } else if (type === 'COMPLETE') {
    audioPlayer.src = URL.createObjectURL(wavBlob);
    audioPlayer.play();
  }
};
```

This completely separated heavy tensor crunching from the UI.

---

### Challenge 2: Bypassing the 29-Second Model Limit

Kokoro-82M has an inherent context window of approximately 512 phonemes (roughly 240 characters or ~29 seconds of speech). If you pass a 2,000-word blog post or book chapter, the model simply truncates it.

To provide truly **unlimited continuous speech**, I built an intelligent chunking and audio concatenation pipeline:

1. **Intelligent Text Chunker:** A regex engine splits text by sentence terminators (`.`, `!`, `?`, `\n`) while preserving natural phrase boundaries (target max chunk size ~200 characters).
2. **Sequential Inference:** Chunks are synthesized sequentially with lightweight progress reporting.
3. **Continuous Audio Stitching:** We insert a natural 220ms pause between sentences and concatenate the raw `Float32Array` PCM audio buffers:

```javascript
// Merging sequential audio buffers with natural pauses
function concatenatePcmBuffers(buffers, sampleRate = 24000) {
  const pauseSamples = Math.floor(sampleRate * 0.22); // 220ms natural pause
  const totalLength = buffers.reduce((acc, b) => acc + b.length + pauseSamples, 0);
  const result = new Float32Array(totalLength);
  
  let offset = 0;
  for (const buffer of buffers) {
    result.set(buffer, offset);
    offset += buffer.length + pauseSamples;
  }
  return result;
}
```

4. **In-Memory WAV Encoder:** Instead of pulling in a heavy external audio library, the worker serializes the merged PCM stream directly into a clean 16-bit PCM WAV Blob with proper RIFF headers on the fly.

---

### Challenge 3: Client-Side Privacy

In enterprise and academic workflows, users often need to read sensitive medical notes, legal documents, or unreleased writing. Because PureTTS downloads the model weights to the browser's local cache (IndexedDB) and computes locally:

* **No text is ever transmitted across the network.**
* **No audio is stored on a server.**
* It works completely offline after the initial model load!

---

### The Result

You can try it live right in your browser here:  
👉 **[https://puretts.com](https://puretts.com)**

* Zero account creation or email logins.
* No character count limits.
* Natural voices with instant WAV download.

I'd love feedback from fellow developers! What features or voices should I add next?
