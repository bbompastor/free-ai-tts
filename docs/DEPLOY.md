# OpenVoice AI — Deployment & Monetization Playbook

## 1. Quick Local Test

Run a local HTTP server inside the `free-ai-tts` folder:

```bash
cd free-ai-tts
python3 -m http.server 8080
```

Open `http://localhost:8080/` in your browser.

---

## 2. Deploy for Free in 3 Minutes

### Option A: Cloudflare Pages (Recommended — Fast & Unlimited Bandwidth)
1. Sign up for a free account at [pages.cloudflare.com](https://pages.cloudflare.com).
2. Click **Create Application** → **Pages** → **Connect to Git** (or drag & drop the `free-ai-tts` folder directly).
3. Build Settings:
   - Framework preset: `None`
   - Build command: `(leave blank)`
   - Build output directory: `.`
4. Click **Deploy**. Your site goes live at `your-project.pages.dev` with a free SSL certificate.
5. *(Optional)* Add your own custom domain (e.g. `freetts.io` or `openvoice.ai`) for $0 extra on Cloudflare.

### Option B: GitHub Pages
1. Create a new GitHub repository (e.g., `free-tts`).
2. Push the contents of the `free-ai-tts` folder:
   ```bash
   git init
   git add .
   git commit -m "Launch OpenVoice AI"
   git branch -M main
   git remote add origin https://github.com/<your-username>/free-tts.git
   git push -u origin main
   ```
3. In your repo on GitHub: Go to **Settings** → **Pages** → Source: **Deploy from a branch** → Branch: `main` / `root`. Click **Save**.
4. Your site will be live at `https://<your-username>.github.io/free-tts/`.

---

## 3. How to Add Ad Revenue (Google AdSense)

1. Apply for an account at [google.com/adsense](https://www.google.com/adsense/).
2. Once approved, Google provides you with an Ad snippet that looks like:
   ```html
   <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=ca-pub-XXXXXXXXXXXX" crossorigin="anonymous"></script>
   ```
3. In `index.html`, replace the two ad placeholder boxes:
   - `<div class="ad-slot" id="ad-top-slot">`
   - `<div class="ad-slot" id="ad-mid-slot">`
   with your Google AdSense unit code.

---

## 4. How to Get Your First 10,000 Users (Free Organic Marketing)

1. **Reddit**:
   - `r/InternetIsBeautiful` ("I made a 100% free, private text-to-speech tool that runs locally in your browser with zero limits")
   - `r/SideProject`
   - `r/freesoftware`
   - `r/voiceacting` & `r/VideoEditing`
2. **Product Hunt**:
   - Submit as a free utility tool on a Tuesday or Wednesday.
3. **TikTok / YouTube Shorts**:
   - Screen record a 15-second video: *"Stop paying $20/month for ElevenLabs. Here's how to generate unlimited AI voices directly inside your browser for free."* Show the text being pasted and playing back.
