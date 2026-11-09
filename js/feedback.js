/**
 * PureTTS Client-Side Feedback Widget & Post-Generation Rating System
 * - 100% static & privacy-first (no third-party form tracking scripts)
 * - Logs structured feedback events to GA4 (gtag)
 * - Generates pre-filled GitHub Issue templates with optional WebGPU/WASM diagnostics
 */
(function () {
  const GITHUB_REPO_ISSUES_URL = "https://github.com/bbompastor/free-ai-tts/issues/new";

  const CATEGORY_META = {
    feature: {
      label: "💡 Feature Request",
      titlePrefix: "[Feature Request]",
      ghLabel: "enhancement",
      placeholder: "What feature, voice, or workflow improvement would make PureTTS better for you?"
    },
    bug: {
      label: "🐛 Bug Report",
      titlePrefix: "[Bug]",
      ghLabel: "bug",
      placeholder: "What happened? (e.g., audio stopped mid-sentence, PDF page extraction issue, WebGPU error...)"
    },
    voice: {
      label: "🎙️ Voice / Audio Quality",
      titlePrefix: "[Voice Feedback]",
      ghLabel: "voice-quality",
      placeholder: "Which voice were you using, and how did the pronunciation, pacing, or audio quality sound?"
    },
    praise: {
      label: "❤️ General Feedback",
      titlePrefix: "[Feedback]",
      ghLabel: "feedback",
      placeholder: "Share how you use PureTTS or any thoughts for the creator!"
    }
  };

  function trackGAEvent(eventName, params = {}) {
    try {
      if (typeof window.gtag === "function") {
        window.gtag("event", eventName, {
          page_path: window.location.pathname,
          ...params
        });
      }
    } catch (e) {
      // Ignore analytics errors if blocked by ad/privacy blockers
    }
  }

  function getStudioDiagnostics() {
    const voiceEl = document.getElementById("voiceSelect");
    const quantEl = document.getElementById("quantSelect");
    const speedEl = document.getElementById("speedRange");
    const textEl = document.getElementById("textInput");
    const hasWebGPU = typeof navigator !== "undefined" && !!navigator.gpu;

    return {
      page: window.location.pathname || "/",
      voice: voiceEl ? voiceEl.value : "n/a",
      engine: quantEl ? quantEl.value : "n/a",
      speed: speedEl ? `${speedEl.value}x` : "n/a",
      charLength: textEl && textEl.value ? textEl.value.trim().length : 0,
      webgpuAvailable: hasWebGPU ? "Yes" : "No",
      userAgent: typeof navigator !== "undefined" ? navigator.userAgent : "Unknown"
    };
  }

  function injectStyles() {
    if (document.getElementById("puretts-feedback-styles")) return;
    const style = document.createElement("style");
    style.id = "puretts-feedback-styles";
    style.textContent = `
      .floating-feedback-btn {
        position: fixed;
        bottom: 24px;
        left: 24px;
        background: rgba(15, 23, 42, 0.92);
        backdrop-filter: blur(12px);
        color: #f8fafc;
        border: 1px solid rgba(139, 92, 246, 0.45);
        padding: 10px 18px;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 0.88rem;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 8px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.5), 0 0 15px rgba(139, 92, 246, 0.25);
        z-index: 100;
        transition: all 0.2s ease;
        font-family: inherit;
      }
      .floating-feedback-btn:hover {
        transform: translateY(-2px);
        background: rgba(139, 92, 246, 0.22);
        border-color: #8b5cf6;
        color: #ffffff;
      }
      @media (max-width: 600px) {
        .floating-feedback-btn {
          bottom: 16px;
          left: 16px;
          padding: 8px 14px;
          font-size: 0.8rem;
        }
      }

      /* Modal Backdrop */
      .fb-modal-backdrop {
        position: fixed;
        inset: 0;
        background: rgba(7, 10, 19, 0.78);
        backdrop-filter: blur(8px);
        z-index: 2000;
        display: none;
        align-items: center;
        justify-content: center;
        padding: 16px;
        opacity: 0;
        transition: opacity 0.2s ease;
      }
      .fb-modal-backdrop.open {
        display: flex;
        opacity: 1;
      }

      /* Modal Card */
      .fb-modal-card {
        background: #0f172a;
        border: 1px solid rgba(139, 92, 246, 0.38);
        border-radius: 16px;
        max-width: 520px;
        width: 100%;
        padding: 24px;
        box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.85), 0 0 30px rgba(139, 92, 246, 0.2);
        color: #f8fafc;
        position: relative;
        max-height: 92vh;
        overflow-y: auto;
      }
      .fb-modal-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        margin-bottom: 14px;
      }
      .fb-modal-title {
        font-size: 1.2rem;
        font-weight: 800;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 8px;
      }
      .fb-modal-subtitle {
        font-size: 0.84rem;
        color: #94a3b8;
        margin-top: 2px;
      }
      .fb-close-btn {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #94a3b8;
        width: 32px;
        height: 32px;
        border-radius: 8px;
        cursor: pointer;
        font-size: 1rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        transition: all 0.15s ease;
      }
      .fb-close-btn:hover {
        color: #ffffff;
        background: rgba(255, 255, 255, 0.14);
      }

      /* Star Rating Row */
      .fb-rating-box {
        background: #090e1a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 12px 16px;
        margin-bottom: 16px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
      }
      .fb-rating-label {
        font-size: 0.85rem;
        font-weight: 600;
        color: #cbd5e1;
      }
      .fb-stars {
        display: inline-flex;
        gap: 6px;
      }
      .fb-star-btn {
        background: transparent;
        border: none;
        font-size: 1.35rem;
        cursor: pointer;
        line-height: 1;
        padding: 2px;
        filter: grayscale(1) opacity(0.45);
        transition: transform 0.15s ease, filter 0.15s ease;
      }
      .fb-star-btn:hover,
      .fb-star-btn.active {
        filter: grayscale(0) opacity(1);
        transform: scale(1.12);
      }

      /* Category Pills */
      .fb-cat-grid {
        display: grid;
        grid-template-columns: repeat(2, 1fr);
        gap: 8px;
        margin-bottom: 14px;
      }
      @media (max-width: 460px) {
        .fb-cat-grid {
          grid-template-columns: 1fr;
        }
      }
      .fb-cat-btn {
        background: #090e1a;
        border: 1px solid #1e293b;
        color: #94a3b8;
        padding: 9px 12px;
        border-radius: 10px;
        font-size: 0.82rem;
        font-weight: 600;
        cursor: pointer;
        text-align: left;
        transition: all 0.15s ease;
        font-family: inherit;
      }
      .fb-cat-btn:hover {
        border-color: rgba(139, 92, 246, 0.5);
        color: #f8fafc;
      }
      .fb-cat-btn.active {
        background: rgba(139, 92, 246, 0.18);
        border-color: #8b5cf6;
        color: #ffffff;
      }

      /* Textarea & Inputs */
      .fb-field-label {
        display: block;
        font-size: 0.82rem;
        font-weight: 600;
        color: #cbd5e1;
        margin-bottom: 6px;
      }
      .fb-textarea {
        width: 100%;
        min-height: 105px;
        padding: 12px;
        background: #090e1a;
        border: 1px solid #1e293b;
        border-radius: 10px;
        color: #f8fafc;
        font-size: 0.9rem;
        line-height: 1.5;
        resize: vertical;
        outline: none;
        margin-bottom: 12px;
        font-family: inherit;
      }
      .fb-textarea:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 3px rgba(139, 92, 246, 0.25);
      }

      .fb-checkbox-row {
        display: flex;
        align-items: center;
        gap: 8px;
        font-size: 0.78rem;
        color: #94a3b8;
        margin-bottom: 16px;
      }
      .fb-checkbox-row input {
        accent-color: #8b5cf6;
        cursor: pointer;
      }
      .fb-checkbox-row label {
        margin: 0;
        font-weight: 400;
        font-size: 0.78rem;
        color: #94a3b8;
        cursor: pointer;
      }

      /* Modal Actions */
      .fb-actions {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
      }
      .fb-btn-primary {
        flex: 1;
        min-width: 180px;
        padding: 11px 16px;
        border-radius: 10px;
        border: none;
        background: linear-gradient(135deg, #8b5cf6, #6d28d9);
        color: #ffffff;
        font-size: 0.88rem;
        font-weight: 700;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        transition: all 0.15s ease;
        font-family: inherit;
      }
      .fb-btn-primary:hover {
        background: linear-gradient(135deg, #9333ea, #581c87);
        transform: translateY(-1px);
      }
      .fb-btn-github {
        padding: 11px 14px;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.16);
        background: rgba(255, 255, 255, 0.05);
        color: #e2e8f0;
        font-size: 0.84rem;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 6px;
        text-decoration: none;
        transition: all 0.15s ease;
        font-family: inherit;
      }
      .fb-btn-github:hover {
        background: rgba(255, 255, 255, 0.12);
        color: #ffffff;
      }

      /* Success Message */
      .fb-success-banner {
        display: none;
        margin-top: 14px;
        padding: 12px 14px;
        background: rgba(16, 185, 129, 0.14);
        border: 1px solid rgba(16, 185, 129, 0.35);
        border-radius: 10px;
        color: #6ee7b7;
        font-size: 0.84rem;
        line-height: 1.45;
      }
      .fb-success-banner.show {
        display: block;
      }

      /* Inline Post-Generation Audio Rating Bar */
      .inline-audio-feedback {
        margin-top: 14px;
        padding: 10px 14px;
        background: rgba(139, 92, 246, 0.07);
        border: 1px solid rgba(139, 92, 246, 0.22);
        border-radius: 10px;
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 10px;
        font-size: 0.83rem;
        color: #cbd5e1;
      }
      .inline-fb-btns {
        display: inline-flex;
        align-items: center;
        gap: 8px;
      }
      .inline-fb-btn {
        background: rgba(15, 23, 42, 0.8);
        border: 1px solid rgba(255, 255, 255, 0.12);
        color: #e2e8f0;
        padding: 5px 12px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        cursor: pointer;
        display: inline-flex;
        align-items: center;
        gap: 5px;
        transition: all 0.15s ease;
        font-family: inherit;
      }
      .inline-fb-btn:hover {
        border-color: #8b5cf6;
        background: rgba(139, 92, 246, 0.2);
        color: #ffffff;
      }
      .inline-fb-btn.voted {
        border-color: #10b981;
        background: rgba(16, 185, 129, 0.18);
        color: #6ee7b7;
      }
    `;
    document.head.appendChild(style);
  }

  let modalBackdrop = null;
  let selectedRating = 5;
  let selectedCategory = "feature";

  function buildGitHubIssueUrl(messageText, includeDiagnostics) {
    const meta = CATEGORY_META[selectedCategory] || CATEGORY_META.feature;
    const summarySnippet = messageText
      ? messageText.split("\n")[0].slice(0, 65)
      : "User feedback on PureTTS";
    const title = `${meta.titlePrefix} ${summarySnippet}`;

    const starsDisplay = "⭐".repeat(selectedRating) + ` (${selectedRating}/5)`;
    const lines = [
      `### Feedback Summary`,
      `- **Category:** ${meta.label}`,
      `- **Rating:** ${starsDisplay}`,
      "",
      `### Details`,
      messageText || "_No additional comment provided._"
    ];

    if (includeDiagnostics) {
      const diag = getStudioDiagnostics();
      lines.push(
        "",
        `### Browser & Studio Environment`,
        `- **Page:** \`${diag.page}\``,
        `- **Selected Voice:** \`${diag.voice}\``,
        `- **Synthesis Engine:** \`${diag.engine}\``,
        `- **Speaking Speed:** \`${diag.speed}\``,
        `- **Text Length:** \`${diag.charLength} chars\``,
        `- **WebGPU Supported:** \`${diag.webgpuAvailable}\``,
        `- **User Agent:** \`${diag.userAgent}\``
      );
    }

    const params = new URLSearchParams({
      title,
      body: lines.join("\n"),
      labels: meta.ghLabel
    });
    return `${GITHUB_REPO_ISSUES_URL}?${params.toString()}`;
  }

  function createFeedbackModal() {
    if (modalBackdrop) return;

    // Floating button
    const floatBtn = document.createElement("button");
    floatBtn.type = "button";
    floatBtn.className = "floating-feedback-btn";
    floatBtn.setAttribute("aria-label", "Send feedback or report an issue");
    floatBtn.innerHTML = `<span>💬</span> <span>Feedback</span>`;
    floatBtn.addEventListener("click", () => openFeedbackModal());
    document.body.appendChild(floatBtn);

    // Modal DOM
    modalBackdrop = document.createElement("div");
    modalBackdrop.className = "fb-modal-backdrop";
    modalBackdrop.setAttribute("role", "dialog");
    modalBackdrop.setAttribute("aria-modal", "true");
    modalBackdrop.setAttribute("aria-labelledby", "fbModalTitle");

    modalBackdrop.innerHTML = `
      <div class="fb-modal-card">
        <div class="fb-modal-header">
          <div>
            <div class="fb-modal-title" id="fbModalTitle">💬 Help Improve PureTTS</div>
            <p class="fb-modal-subtitle">100% private &amp; open-source. Share a feature idea, voice rating, or bug report.</p>
          </div>
          <button type="button" class="fb-close-btn" id="fbCloseModalBtn" aria-label="Close feedback modal">✕</button>
        </div>

        <div class="fb-rating-box">
          <span class="fb-rating-label">How is your experience with PureTTS?</span>
          <div class="fb-stars" id="fbStarsGroup" role="radiogroup" aria-label="Star rating">
            <button type="button" class="fb-star-btn active" data-star="1" aria-label="1 star">⭐</button>
            <button type="button" class="fb-star-btn active" data-star="2" aria-label="2 stars">⭐</button>
            <button type="button" class="fb-star-btn active" data-star="3" aria-label="3 stars">⭐</button>
            <button type="button" class="fb-star-btn active" data-star="4" aria-label="4 stars">⭐</button>
            <button type="button" class="fb-star-btn active" data-star="5" aria-label="5 stars">⭐</button>
          </div>
        </div>

        <div class="fb-cat-grid" id="fbCategoryGrid">
          <button type="button" class="fb-cat-btn active" data-cat="feature">💡 Feature Request</button>
          <button type="button" class="fb-cat-btn" data-cat="voice">🎙️ Voice / Audio Quality</button>
          <button type="button" class="fb-cat-btn" data-cat="bug">🐛 Report a Bug</button>
          <button type="button" class="fb-cat-btn" data-cat="praise">❤️ General Feedback</button>
        </div>

        <label class="fb-field-label" for="fbMessageInput">Your Feedback or Details</label>
        <textarea id="fbMessageInput" class="fb-textarea" placeholder="${CATEGORY_META.feature.placeholder}"></textarea>

        <div class="fb-checkbox-row">
          <input type="checkbox" id="fbDiagCheckbox" checked />
          <label for="fbDiagCheckbox">Include non-personal browser &amp; synthesis engine info (WebGPU/WASM, voice, speed)</label>
        </div>

        <div class="fb-actions">
          <button type="button" class="fb-btn-primary" id="fbSubmitQuickBtn">
            <span>⚡</span> <span>Send Quick Feedback</span>
          </button>
          <button type="button" class="fb-btn-github" id="fbOpenGithubBtn" title="Opens a pre-filled issue on GitHub">
            <span>🐙</span> <span>Post on GitHub Issues ↗</span>
          </button>
        </div>

        <div class="fb-success-banner" id="fbSuccessBanner"></div>
      </div>
    `;

    document.body.appendChild(modalBackdrop);

    // Wire events
    const closeBtn = modalBackdrop.querySelector("#fbCloseModalBtn");
    const stars = modalBackdrop.querySelectorAll(".fb-star-btn");
    const catBtns = modalBackdrop.querySelectorAll(".fb-cat-btn");
    const msgInput = modalBackdrop.querySelector("#fbMessageInput");
    const diagCheckbox = modalBackdrop.querySelector("#fbDiagCheckbox");
    const submitQuickBtn = modalBackdrop.querySelector("#fbSubmitQuickBtn");
    const openGithubBtn = modalBackdrop.querySelector("#fbOpenGithubBtn");
    const successBanner = modalBackdrop.querySelector("#fbSuccessBanner");

    closeBtn.addEventListener("click", closeFeedbackModal);
    modalBackdrop.addEventListener("click", (e) => {
      if (e.target === modalBackdrop) closeFeedbackModal();
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && modalBackdrop.classList.contains("open")) {
        closeFeedbackModal();
      }
    });

    stars.forEach((btn) => {
      btn.addEventListener("click", () => {
        selectedRating = parseInt(btn.dataset.star, 10) || 5;
        stars.forEach((s) => {
          const val = parseInt(s.dataset.star, 10);
          s.classList.toggle("active", val <= selectedRating);
        });
        trackGAEvent("feedback_star_select", { rating: selectedRating });
      });
    });

    catBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        selectedCategory = btn.dataset.cat;
        catBtns.forEach((b) => b.classList.toggle("active", b === btn));
        const meta = CATEGORY_META[selectedCategory];
        if (meta && msgInput) {
          msgInput.placeholder = meta.placeholder;
        }
      });
    });

    submitQuickBtn.addEventListener("click", () => {
      const messageText = msgInput.value.trim();
      const diag = getStudioDiagnostics();

      trackGAEvent("feedback_submit", {
        category: selectedCategory,
        rating: selectedRating,
        voice: diag.voice,
        engine: diag.engine,
        webgpu: diag.webgpuAvailable,
        feedback_text: messageText.slice(0, 100) || "(rating_only)"
      });

      // Save locally in localStorage so user knows it was recorded
      try {
        const existing = JSON.parse(localStorage.getItem("puretts_feedback_log") || "[]");
        existing.push({
          ts: new Date().toISOString(),
          rating: selectedRating,
          category: selectedCategory,
          message: messageText,
          page: diag.page
        });
        localStorage.setItem("puretts_feedback_log", JSON.stringify(existing.slice(-20)));
      } catch (e) {
        // ignore storage errors
      }

      const ghUrl = buildGitHubIssueUrl(messageText, diagCheckbox.checked);
      successBanner.innerHTML = `
        <strong>✓ Thank you for your feedback!</strong> Your rating &amp; note have been recorded.
        ${
          messageText
            ? `<div style="margin-top:6px;">Want a direct reply from the developer or public tracking? <a href="${ghUrl}" target="_blank" rel="noopener" style="color:#ffffff;text-decoration:underline;font-weight:700;">Submit this pre-filled report on GitHub Issues →</a></div>`
            : ""
        }
      `;
      successBanner.classList.add("show");
    });

    openGithubBtn.addEventListener("click", () => {
      const messageText = msgInput.value.trim();
      const diag = getStudioDiagnostics();
      trackGAEvent("feedback_github_open", {
        category: selectedCategory,
        rating: selectedRating,
        voice: diag.voice,
        engine: diag.engine
      });
      const ghUrl = buildGitHubIssueUrl(messageText, diagCheckbox.checked);
      window.open(ghUrl, "_blank", "noopener,noreferrer");
    });
  }

  function openFeedbackModal(options = {}) {
    createFeedbackModal();
    const catBtns = modalBackdrop.querySelectorAll(".fb-cat-btn");
    const msgInput = modalBackdrop.querySelector("#fbMessageInput");
    const successBanner = modalBackdrop.querySelector("#fbSuccessBanner");

    if (options.category && CATEGORY_META[options.category]) {
      selectedCategory = options.category;
      catBtns.forEach((b) => b.classList.toggle("active", b.dataset.cat === selectedCategory));
      msgInput.placeholder = CATEGORY_META[selectedCategory].placeholder;
    }

    if (typeof options.rating === "number") {
      selectedRating = options.rating;
      const stars = modalBackdrop.querySelectorAll(".fb-star-btn");
      stars.forEach((s) => {
        const val = parseInt(s.dataset.star, 10);
        s.classList.toggle("active", val <= selectedRating);
      });
    }

    successBanner.classList.remove("show");
    modalBackdrop.classList.add("open");
    trackGAEvent("feedback_modal_open", { trigger: options.trigger || "floating_button" });
    setTimeout(() => msgInput.focus(), 80);
  }

  function closeFeedbackModal() {
    if (!modalBackdrop) return;
    modalBackdrop.classList.remove("open");
  }

  function injectPostGenerationBar() {
    const audioResult = document.getElementById("audioResult");
    if (!audioResult || document.getElementById("inlineAudioFeedback")) return;

    const bar = document.createElement("div");
    bar.id = "inlineAudioFeedback";
    bar.className = "inline-audio-feedback";
    bar.innerHTML = `
      <span id="inlineFbPrompt">🎧 How did this generated voiceover sound?</span>
      <div class="inline-fb-btns">
        <button type="button" class="inline-fb-btn" id="inlineFbUpBtn" aria-label="Audio sounded great">👍 Great</button>
        <button type="button" class="inline-fb-btn" id="inlineFbDownBtn" aria-label="Report audio issue or suggest improvement">👎 Needs Work / Feedback</button>
      </div>
    `;

    audioResult.appendChild(bar);

    const upBtn = bar.querySelector("#inlineFbUpBtn");
    const downBtn = bar.querySelector("#inlineFbDownBtn");
    const promptEl = bar.querySelector("#inlineFbPrompt");

    upBtn.addEventListener("click", () => {
      const diag = getStudioDiagnostics();
      trackGAEvent("tts_audio_feedback", {
        vote: "up",
        voice: diag.voice,
        engine: diag.engine,
        speed: diag.speed
      });
      upBtn.classList.add("voted");
      promptEl.innerHTML = `💜 Glad it sounded great! <a href="#" id="inlineLeaveNoteLink" style="color:#c4b5fd;text-decoration:underline;margin-left:4px;">Leave a comment or feature idea →</a>`;
      const noteLink = bar.querySelector("#inlineLeaveNoteLink");
      if (noteLink) {
        noteLink.addEventListener("click", (e) => {
          e.preventDefault();
          openFeedbackModal({ category: "praise", rating: 5, trigger: "inline_up_note" });
        });
      }
    });

    downBtn.addEventListener("click", () => {
      const diag = getStudioDiagnostics();
      trackGAEvent("tts_audio_feedback", {
        vote: "down",
        voice: diag.voice,
        engine: diag.engine,
        speed: diag.speed
      });
      openFeedbackModal({ category: "voice", rating: 3, trigger: "inline_down" });
    });
  }

  function enhanceFooterAndTriggers() {
    // Bind any explicit [data-feedback-trigger] elements on the page
    document.querySelectorAll("[data-feedback-trigger]").forEach((el) => {
      el.addEventListener("click", (e) => {
        e.preventDefault();
        const cat = el.getAttribute("data-feedback-category") || "feature";
        openFeedbackModal({ category: cat, trigger: "page_trigger" });
      });
    });

    // Add a "💬 Send Feedback" link in .footer-links if not already present
    document.querySelectorAll(".footer-links").forEach((footerNav) => {
      if (footerNav.querySelector("[data-feedback-footer]")) return;
      const fbLink = document.createElement("a");
      fbLink.href = "#feedback";
      fbLink.setAttribute("data-feedback-footer", "true");
      fbLink.textContent = "💬 Send Feedback";
      fbLink.addEventListener("click", (e) => {
        e.preventDefault();
        openFeedbackModal({ trigger: "footer_link" });
      });
      footerNav.appendChild(fbLink);
    });
  }

  function init() {
    injectStyles();
    createFeedbackModal();
    injectPostGenerationBar();
    enhanceFooterAndTriggers();
    window.openPureTTSFeedback = openFeedbackModal;
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
