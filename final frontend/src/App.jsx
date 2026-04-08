/**
 * SnapWise — Frontend (React + Vite)
 * ═══════════════════════════════════════════════════════════════════
 *
 * KEY FEATURES IN THIS FILE
 * ──────────────────────────
 * 1. FileDropzone        — drag-and-drop / click file upload widget
 * 2. KeywordHighlighter  — draws yellow keyword highlights on the ORIGINAL
 *                          image using HTML Canvas API, with a download button
 * 3. ImageComparison     — side-by-side: CV-preprocessed image only
 *                          (original is shown in KeywordHighlighter above)
 * 4. AnalysisBadges      — metric cards: topic, confidence, readability, words
 * 5. KeywordCloud        — TF-IDF keyword pills (text list)
 * 6. ResultCard          — heading, summary, OCR text, visual elements
 * 7. LoadingPulse        — animated loading indicator
 * 8. App (root)          — state management + API calls
 *
 * DATA FLOW
 * ──────────
 *   User uploads image
 *     → POST /api/analyse/image
 *     → backend returns:
 *         original_image_b64      ← raw original (used for Canvas highlight)
 *         original_image_mime     ← MIME type for data-URL construction
 *         preprocessed_image_b64  ← CV-cleaned image (side-by-side preview)
 *         keywords[]              ← TF-IDF extracted terms
 *         topic, confidence       ← BART-MNLI zero-shot classification
 *         readability             ← Flesch Reading Ease on OCR text
 *         heading, summary        ← from Qwen2.5-VL
 *         extracted_text          ← raw OCR output
 */

import { useState, useRef, useCallback, useEffect } from "react";
import "./App.css";

// ─── API base URL — Flask backend ────────────────────────────────────────────
const API_BASE = "http://localhost:5000/api";


// ═════════════════════════════════════════════════════════════════════════════
// 1. FileDropzone
//    Handles both drag-and-drop and click-to-browse file selection.
//    Calls onFile(file) when a file is chosen.
// ═════════════════════════════════════════════════════════════════════════════
function FileDropzone({ onFile, accept, label, icon, disabled }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

  // handleDrop — fires when user drops a file onto the zone
  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setDragging(false);
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  }, [onFile]);

  return (
    <div
      className={`dropzone ${dragging ? "dragging" : ""} ${disabled ? "disabled" : ""}`}
      onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={handleDrop}
      onClick={() => !disabled && inputRef.current.click()}
    >
      {/* Hidden real file input — triggered by clicking the styled div */}
      <input
        ref={inputRef}
        type="file"
        accept={accept}
        style={{ display: "none" }}
        onChange={(e) => e.target.files[0] && onFile(e.target.files[0])}
      />
      <div className="dropzone-icon">{icon}</div>
      <p className="dropzone-label">{label}</p>
      <p className="dropzone-hint">drag & drop or click to browse</p>
    </div>
  );
}



// ═════════════════════════════════════════════════════════════════════════════
// 2. KeywordHighlighter
//
//    Draws PRECISE yellow highlight boxes on the ORIGINAL image using
//    exact Tesseract bounding boxes returned by the backend.
//
//    How it works:
//    ─────────────
//    • Backend runs pytesseract.image_to_data() on the original image
//      → returns word_boxes: [{keyword, x, y, w, h}] in exact pixel coords
//    • We load the original image onto an HTML <canvas>
//    • For each box we draw:
//        ① Semi-transparent yellow fill  rgba(255, 220, 0, 0.40)
//        ② Amber border                  rgba(180, 130, 0, 0.85)
//        ③ Small keyword pill label above the box
//    • Every highlight is EXACTLY around the matched word — no heuristics.
//    • Download button: canvas.toDataURL() → hidden <a>.click() → PNG save
// ═════════════════════════════════════════════════════════════════════════════
function KeywordHighlighter({ originalB64, originalMime, wordBoxes, filename }) {
  const canvasRef = useRef(null);

  // ── Re-draw whenever image or bounding boxes change ──────────────────────
  useEffect(() => {
    if (!originalB64 || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    const img    = new Image();

    // Build data-URL: "data:<mime>;base64,<b64string>"
    img.src = `data:${originalMime || "image/png"};base64,${originalB64}`;

    img.onload = () => {
      // ── Step 1: Fit canvas to image natural size ─────────────────────────
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // ── Step 2: Paint original image as background ────────────────────────
      ctx.drawImage(img, 0, 0);

      // No boxes → just show the image, nothing to highlight
      if (!wordBoxes || wordBoxes.length === 0) return;

      // Font size scales with image width so labels look good at any resolution
      const fontSize = Math.max(10, Math.min(18, canvas.width / 55));

      // ── Step 3: Draw each Tesseract bounding box ─────────────────────────
      wordBoxes.forEach(({ keyword, x, y, w, h }) => {
        if (w <= 0 || h <= 0) return;  // skip degenerate boxes (Tesseract noise)

        // ① Yellow semi-transparent fill
        //    Bright enough to see, transparent enough to read the underlying text
        ctx.fillStyle = "rgba(255, 220, 0, 0.40)";
        ctx.fillRect(x, y, w, h);

        // ② Amber border — slightly darker than fill for crisp edges
        ctx.strokeStyle = "rgba(180, 130, 0, 0.85)";
        ctx.lineWidth   = Math.max(1.5, canvas.width / 800);
        ctx.strokeRect(x, y, w, h);

        // ③ Keyword pill label positioned ABOVE the bounding box
        ctx.font = `bold ${fontSize}px sans-serif`;
        const padX  = 6;
        const padY  = 4;
        const textW = ctx.measureText(keyword).width;
        const pillW = textW + padX * 2;
        const pillH = fontSize + padY * 2;

        // Clamp pill X so it never overflows right edge
        const pillX = Math.max(0, Math.min(x, canvas.width - pillW));
        // Clamp pill Y so it never goes above top of canvas
        const pillY = Math.max(0, y - pillH - 2);

        // Pill background: solid yellow
        ctx.fillStyle = "rgba(255, 200, 0, 0.95)";
        if (ctx.roundRect) {
          ctx.beginPath();
          ctx.roundRect(pillX, pillY, pillW, pillH, 4);
          ctx.fill();
        } else {
          ctx.fillRect(pillX, pillY, pillW, pillH);   // fallback for older browsers
        }

        // Pill text: dark brown — high contrast on yellow background
        ctx.fillStyle = "#3d2800";
        ctx.fillText(keyword, pillX + padX, pillY + fontSize + padY - 2);
      });

      // ── Step 4: Legend badge in top-right corner ─────────────────────────
      const legendTxt = `${wordBoxes.length} keyword match${wordBoxes.length !== 1 ? "es" : ""}`;
      ctx.font = `${Math.max(11, fontSize - 2)}px sans-serif`;
      const legW = ctx.measureText(legendTxt).width + 16;
      const legH = fontSize + 10;
      const legX = canvas.width - legW - 10;
      const legY = 10;

      // Dark pill for legend
      ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
      if (ctx.roundRect) {
        ctx.beginPath(); ctx.roundRect(legX, legY, legW, legH, 5); ctx.fill();
      } else {
        ctx.fillRect(legX, legY, legW, legH);
      }
      ctx.fillStyle = "#ffe066";
      ctx.fillText(legendTxt, legX + 8, legY + fontSize + 2);
    };
  }, [originalB64, originalMime, wordBoxes]);

  // ── Download handler ──────────────────────────────────────────────────────
  // canvas.toDataURL("image/png") encodes the FULL canvas (original image +
  // all drawn highlights) as a base64 PNG string.
  // We create a hidden <a>, set href to that string, and click it —
  // browser saves the file with no server round-trip needed.
  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const url  = canvas.toDataURL("image/png");
    const link = document.createElement("a");
    // Build filename: strip extension, append suffix
    const base = (filename || "image").replace(/\.[^/.]+$/, "");
    link.download = `${base}_keywords_highlighted.png`;
    link.href     = url;
    link.click();
  };

  if (!originalB64) return null;

  return (
    <div className="highlight-container">

      {/* Row: title on left, download button on right */}
      <div className="highlight-header">
        <span className="highlight-title">
          🟡 Original Image — Keyword Highlights (exact Tesseract bounding boxes)
        </span>
        {/* Download button triggers canvas → PNG save */}
        <button className="download-btn" onClick={handleDownload}
          title="Download image with keyword highlights as PNG">
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Download Highlighted
        </button>
      </div>

      {/* Canvas — bounding boxes drawn here by useEffect above */}
      <canvas ref={canvasRef} className="highlight-canvas"
        title="Yellow = exact Tesseract word bounding box for each keyword" />

      {/* Status note */}
      {(!wordBoxes || wordBoxes.length === 0) && (
        <p className="highlight-note" style={{ color: "var(--text-muted)" }}>
          No keyword bounding boxes found. Ensure Tesseract is installed on the server.
        </p>
      )}
      {wordBoxes && wordBoxes.length > 0 && (
        <p className="highlight-note">
          ✦ Highlights use exact Tesseract OCR word coordinates — perfectly sized bounding boxes
        </p>
      )}
    </div>
  );
}



// ═════════════════════════════════════════════════════════════════════════════
// 3. ImageComparison
//    Shows the CV-preprocessed image (after OpenCV pipeline).
//    The ORIGINAL is shown in KeywordHighlighter above — so this section
//    only shows the "after" side for the CV pipeline comparison.
// ═════════════════════════════════════════════════════════════════════════════
function ImageComparison({ preprocessedB64 }) {
  if (!preprocessedB64) return null;

  return (
    <div className="img-comparison">
      <div className="img-box">
        {/* Label */}
        <span className="img-label preprocessed">CV Preprocessed Output</span>
        <img
          src={`data:image/png;base64,${preprocessedB64}`}
          alt="CV Preprocessed"
          className="compare-img"
        />
        {/* Pipeline steps caption */}
        <span className="img-caption">
          upscale → deskew → gaussian denoise → CLAHE → unsharp mask → adaptive threshold → morph close
        </span>
      </div>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════════
// 4. AnalysisBadges
//    Displays ML/DL analysis results as a grid of metric cards.
//    Fields:
//      topic       — from BART-MNLI zero-shot classification (DL)
//      confidence  — model's confidence score (0-1)
//      readability — Flesch Reading Ease score + level (computed on OCR TEXT)
//      word_count  — raw word count from OCR output
// ═════════════════════════════════════════════════════════════════════════════
function AnalysisBadges({ topic, confidence, readability, word_count }) {
  if (!topic) return null;

  const confPct = confidence ? Math.round(confidence * 100) : 0;

  // Readability colour coding:
  //   green  (≥60) = easy to read
  //   amber  (≥40) = moderate
  //   red    (<40) = complex / academic
  const readColor =
    (readability?.score ?? 0) >= 60 ? "var(--accent3)" :
    (readability?.score ?? 0) >= 40 ? "#f59e0b" : "#f87171";

  return (
    <div className="analysis-grid">

      {/* Zero-shot DL topic result */}
      <div className="analysis-card">
        <span className="ac-label">Topic (Zero-shot DL)</span>
        <span className="ac-value topic">{topic}</span>
        <span className="ac-sub">BART-MNLI · {confPct}% confident</span>
      </div>

      {/* Flesch Reading Ease — computed on OCR text, NOT the image */}
      <div className="analysis-card">
        <span className="ac-label">Readability (OCR text)</span>
        <span className="ac-value" style={{ color: readColor }}>
          {readability?.score ?? "—"}
        </span>
        <span className="ac-sub">{readability?.level ?? ""}</span>
      </div>

      {/* Confidence bar — visual representation of DL model certainty */}
      <div className="analysis-card">
        <span className="ac-label">Model Confidence</span>
        <div className="conf-bar-wrap">
          <div className="conf-bar" style={{ width: `${confPct}%` }} />
        </div>
        <span className="ac-sub">{confPct}%</span>
      </div>

      {/* Word count from OCR */}
      <div className="analysis-card">
        <span className="ac-label">Word Count</span>
        <span className="ac-value">{word_count ?? "—"}</span>
        <span className="ac-sub">from OCR output</span>
      </div>

    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════════
// 5. KeywordCloud
//    Renders TF-IDF keywords as styled pill badges (text list version).
//    Pill size decreases slightly by rank — rank-1 keyword is slightly larger.
//    These are the same keywords visualised on the image in KeywordHighlighter.
// ═════════════════════════════════════════════════════════════════════════════
function KeywordCloud({ keywords }) {
  if (!keywords || keywords.length === 0) return null;

  return (
    <div className="keyword-section">
      <span className="section-label">🔑 TF-IDF Keywords (ranked by score)</span>
      <div className="keyword-cloud">
        {keywords.map((kw, i) => (
          <span
            key={i}
            className="kw-pill"
            // Higher-ranked keywords get slightly larger font
            style={{ fontSize: `${Math.max(0.72, 0.9 - i * 0.02)}rem` }}
          >
            {kw}
          </span>
        ))}
      </div>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════════
// 6. ResultCard
//    Shows the VLM's structured output: heading, summary, OCR text,
//    visual elements, plus the ML/DL analysis badges and keyword cloud.
// ═════════════════════════════════════════════════════════════════════════════
function ResultCard({ result, pageNum, filename }) {
  const [ocrExpanded, setOcrExpanded] = useState(false);

  return (
    <div className="result-card">

      {/* Page badge — only shown in PDF mode */}
      {pageNum && <div className="page-badge">Page {pageNum}</div>}

      {/* ── Keyword-highlighted ORIGINAL image + download button ─────────── */}
      {/* This is the KEY visual feature: original image with yellow highlights */}
      {/* word_boxes = exact Tesseract pixel bounding boxes from backend */}
      {/* Each box: { keyword, x, y, w, h } in original image pixel space */}
      <KeywordHighlighter
        originalB64={result.original_image_b64}
        originalMime={result.original_image_mime}
        wordBoxes={result.word_boxes}
        filename={filename}
      />

      {/* ── VLM-generated heading ─────────────────────────────────────────── */}
      <h2 className="result-heading">{result.heading}</h2>

      {/* ── ML/DL metric badges ───────────────────────────────────────────── */}
      <AnalysisBadges
        topic={result.topic}
        confidence={result.confidence}
        readability={result.readability}
        word_count={result.word_count}
      />

      {/* ── TF-IDF keyword pills (text form) ─────────────────────────────── */}
      <KeywordCloud keywords={result.keywords} />

      {/* ── VLM summary ───────────────────────────────────────────────────── */}
      <div className="result-section">
        <span className="section-label">📝 Summary</span>
        <p className="result-summary">{result.summary}</p>
      </div>

      {/* ── Visual elements description from VLM ──────────────────────────── */}
      {result.visual_elements && result.visual_elements !== "None" && (
        <div className="result-section">
          <span className="section-label">🖼️ Visual Elements</span>
          <p className="result-visual">{result.visual_elements}</p>
        </div>
      )}

      {/* ── CV preprocessed image ─────────────────────────────────────────── */}
      {result.preprocessed_image_b64 && (
        <div className="result-section">
          <span className="section-label">🔬 CV Preprocessed Image</span>
          <ImageComparison preprocessedB64={result.preprocessed_image_b64} />
        </div>
      )}

      {/* ── Raw OCR text — collapsed by default ───────────────────────────── */}
      {result.extracted_text && (
        <div className="result-section">
          <button
            className="expand-btn"
            onClick={() => setOcrExpanded(!ocrExpanded)}
          >
            {ocrExpanded ? "▲ Hide" : "▼ Show"} Raw OCR Text
          </button>
          {ocrExpanded && (
            <pre className="extracted-text">{result.extracted_text}</pre>
          )}
        </div>
      )}

    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════════
// 7. LoadingPulse
//    Animated concentric rings with a cycling status message.
//    Messages describe the actual pipeline stage running on the backend.
// ═════════════════════════════════════════════════════════════════════════════
function LoadingPulse({ message }) {
  return (
    <div className="loading-container">
      <div className="pulse-ring" />
      <div className="pulse-ring delay1" />
      <div className="pulse-ring delay2" />
      <p className="loading-text">{message}</p>
    </div>
  );
}


// ═════════════════════════════════════════════════════════════════════════════
// 8. App — root component
//    Manages all application state and triggers the API call.
//    State:
//      mode       — "image" | "pdf"
//      file       — the selected File object
//      preview    — object URL of the original image (for upload preview)
//      loading    — boolean: true while waiting for backend
//      loadingMsg — current pipeline stage message shown in LoadingPulse
//      result     — parsed JSON response from the backend
//      error      — error message string
// ═════════════════════════════════════════════════════════════════════════════
export default function App() {
  const [mode,       setMode]       = useState("image");
  const [file,       setFile]       = useState(null);
  const [preview,    setPreview]    = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [result,     setResult]     = useState(null);
  const [error,      setError]      = useState(null);

  // handleFile — called when user picks a file from the dropzone
  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    // Create a local object URL for the original image upload preview
    if (f.type.startsWith("image/")) {
      setPreview(URL.createObjectURL(f));
    } else {
      setPreview(null);
    }
  };

  // handleAnalyse — submits file to backend, cycles loading messages
  const handleAnalyse = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    const endpoint = mode === "image" ? "/analyse/image" : "/analyse/pdf";

    // Loading messages match the actual backend pipeline stages
    const msgs = mode === "image"
      ? [
          "Reading original image...",
          "Running CV preprocessing pipeline...",
          "Deskewing & denoising...",
          "CLAHE contrast enhancement...",
          "Sending to Qwen2.5-VL (OCR + captioning)...",
          "Running TF-IDF keyword extraction...",
          "Zero-shot topic classification (BART-MNLI)...",
          "Computing Flesch readability score...",
          "Building your study notes...",
        ]
      : [
          "Rasterising PDF pages (PyMuPDF)...",
          "Running CV pipeline on each page...",
          "OCR with Qwen2.5-VL...",
          "TF-IDF keyword extraction...",
          "Zero-shot topic classification...",
          "Generating overall document summary...",
        ];

    let msgIdx = 0;
    setLoadingMsg(msgs[0]);
    const msgInterval = setInterval(() => {
      msgIdx = (msgIdx + 1) % msgs.length;
      setLoadingMsg(msgs[msgIdx]);
    }, 2500);

    try {
      const res  = await fetch(`${API_BASE}${endpoint}`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok || !data.success) throw new Error(data.error || "Analysis failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      clearInterval(msgInterval);
      setLoading(false);
    }
  };

  // reset — clears all state to allow a fresh upload
  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      {/* ── Animated background orbs (pure CSS, decorative) ──────────────── */}
      <div className="bg-orb orb1" />
      <div className="bg-orb orb2" />
      <div className="bg-orb orb3" />

      {/* ── Header ─────────────────────────────────────────────────────────── */}
      <header className="header">
        <div className="logo">
          <span className="logo-icon">◈</span>
          <span className="logo-text">SnapWise</span>
        </div>
        <p className="tagline">
          OpenCV · Qwen2.5-VL · TF-IDF · BART-MNLI · Flesch Readability
        </p>
      </header>

      <main className="main">

        {/* ── Mode toggle: Screenshot vs PDF ──────────────────────────────── */}
        <div className="mode-toggle">
          <button
            className={`mode-btn ${mode === "image" ? "active" : ""}`}
            onClick={() => { setMode("image"); reset(); }}
          >
            🖼️ Screenshot
          </button>
          <button
            className={`mode-btn ${mode === "pdf" ? "active" : ""}`}
            onClick={() => { setMode("pdf"); reset(); }}
          >
            📄 PDF Document
          </button>
        </div>

        {/* ── Upload section — shown only before results are ready ─────────── */}
        {!result && (
          <section className="upload-section">
            <FileDropzone
              onFile={handleFile}
              accept={mode === "image" ? "image/*" : "application/pdf"}
              label={mode === "image" ? "Upload a Screenshot" : "Upload a PDF"}
              icon={mode === "image" ? "🖼️" : "📄"}
              disabled={loading}
            />

            {/* File info row */}
            {file && (
              <div className="file-info">
                <span className="file-name">✓ {file.name}</span>
                <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
              </div>
            )}

            {/* Original image upload preview */}
            {preview && (
              <div className="preview-container">
                <img src={preview} alt="Upload preview" className="preview-img" />
              </div>
            )}

            {/* Analyse button — only shows after file is selected */}
            {file && !loading && (
              <button className="analyse-btn" onClick={handleAnalyse}>
                <span>Analyse with AI</span>
                <span className="btn-arrow">→</span>
              </button>
            )}
          </section>
        )}

        {/* ── Loading spinner ──────────────────────────────────────────────── */}
        {loading && <LoadingPulse message={loadingMsg} />}

        {/* ── Error display ────────────────────────────────────────────────── */}
        {error && (
          <div className="error-box">
            <span>⚠️ {error}</span>
            <button onClick={reset} className="retry-btn">Try Again</button>
          </div>
        )}

        {/* ── Results section ──────────────────────────────────────────────── */}
        {result && !loading && (
          <section className="results-section">
            <div className="results-header">
              <h1 className="results-title">Analysis Complete</h1>
              <button className="new-btn" onClick={reset}>+ New Analysis</button>
            </div>

            {/* ── IMAGE result ─────────────────────────────────────────────── */}
            {result.type === "image" && (
              <ResultCard
                result={result.result}
                filename={file?.name}
              />
            )}

            {/* ── PDF result ───────────────────────────────────────────────── */}
            {result.type === "pdf" && (
              <>
                <div className="pdf-meta">
                  📑 {result.total_pages} pages total · {result.analysed_pages} analysed
                </div>

                {/* Overall document summary card (no per-page image here) */}
                <div className="result-card">
                  <h2 className="result-heading">{result.overall.heading}</h2>
                  <AnalysisBadges
                    topic={result.overall.topic}
                    confidence={result.overall.confidence}
                    readability={result.overall.readability}
                    word_count={result.overall.word_count}
                  />
                  <KeywordCloud keywords={result.overall.keywords} />
                  <div className="result-section">
                    <span className="section-label">📝 Document Summary</span>
                    <p className="result-summary">{result.overall.summary}</p>
                  </div>
                </div>

                <div className="pages-divider"><span>Per-Page Breakdown</span></div>

                {/* Each page — with keyword highlight + download */}
                {result.pages.map((page) => (
                  <ResultCard
                    key={page.page}
                    result={page}
                    pageNum={page.page}
                    filename={`${file?.name || "page"}_p${page.page}`}
                  />
                ))}
              </>
            )}
          </section>
        )}
      </main>

      {/* ── Footer ─────────────────────────────────────────────────────────── */}
      <footer className="footer">
        <p>
          CV: OpenCV · VLM: Qwen2.5-VL · ML: TF-IDF · DL: BART-MNLI · Readability: Flesch
          <br />SnapWise — built by Shlok · 2026
        </p>
      </footer>
    </div>
  );
}
