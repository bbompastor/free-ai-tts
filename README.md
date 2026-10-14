# 🎙️ OpenVoice AI — Free Unlimited Text to Speech

> A 100% free, private, unlimited Text-to-Speech (TTS) converter that runs directly in your browser using **WebAssembly** and **WebGPU**. Zero server costs, zero subscriptions, zero character limits.

---

## ⚡ Highlights

* **100% Client-Side Privacy**: Neural AI inference runs in your browser via WebGPU / WebAssembly. Text and audio never leave your device.
* **Unlimited Text Length**: Automatic sentence-level chunking and continuous WAV concatenation. No 250-character paywalls.
* **Studio Quality Voices**: Powered by the open-weight **Kokoro-82M** neural model.
* **Zero UI Freezing**: Offloaded entirely to an isolated background **Web Worker** (`tts.worker.js`), maintaining a smooth 60 FPS interface.
* **$0 Hosting Bill**: Deployable to **GitHub Pages** or **Cloudflare Pages** for free.
* **Ad-Ready**: Pre-configured layout slots for Google AdSense or EthicalAds.

---

## 🚀 Quick Start (Local)

Run a local HTTP server:

```bash
# Python 3
python3 -m http.server 8080

# Or Node.js
npx serve .
```

Open `http://localhost:8080/` in your browser.

---

## 🌐 Deploy to GitHub Pages (100% Free)

1. Create a new repository on your GitHub account: `https://github.com/new` named `free-ai-tts` (or any name you like).
2. Push this repository to GitHub:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: OpenVoice AI Web App"
   git branch -M main
   git remote add origin https://github.com/bbompastor/free-ai-tts.git
   git push -u origin main
   ```
3. In your GitHub repository:
   * Go to **Settings** → **Pages**.
   * Under **Build and deployment** > **Source**, choose **Deploy from a branch**.
   * Select branch: **`main`** and folder: **`/ (root)`**.
   * Click **Save**.
4. Your site will be live at:  
   👉 `https://bbompastor.github.io/free-ai-tts/`

---

## 📁 Project Structure

```
├── index.html       # Main SEO-optimized web application & UI
├── tts.worker.js    # Background Web Worker with sentence chunker & WAV encoder
├── privacy.html     # Google AdSense-compliant privacy policy
├── terms.html       # Terms of Service & Apache 2.0 commercial rights
├── DEPLOY.md        # Comprehensive deployment & monetization playbook
└── README.md        # Project documentation
```

---

## 📜 License

* Code: MIT License
* Kokoro-82M Model: Apache 2.0 License (Commercially permissive)
