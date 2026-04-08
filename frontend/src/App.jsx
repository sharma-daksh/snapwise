import { useState, useRef, useCallback, useEffect } from "react";
import "./App.css";

const API_BASE = "http://localhost:5000/api";


function FileDropzone({ onFile, accept, label, icon, disabled }) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef();

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


function KeywordHighlighter({ originalB64, originalMime, wordBoxes, filename }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    if (!originalB64 || !canvasRef.current) return;

    const canvas = canvasRef.current;
    const ctx    = canvas.getContext("2d");
    const img    = new Image();
    img.src      = `data:${originalMime || "image/png"};base64,${originalB64}`;

    img.onload = () => {
      canvas.width  = img.naturalWidth;
      canvas.height = img.naturalHeight;
      ctx.drawImage(img, 0, 0);

      if (!wordBoxes || wordBoxes.length === 0) return;

      const fontSize = Math.max(10, Math.min(18, canvas.width / 55));

      wordBoxes.forEach(({ keyword, x, y, w, h }) => {
        if (w <= 0 || h <= 0) return;

        ctx.fillStyle = "rgba(255, 220, 0, 0.40)";
        ctx.fillRect(x, y, w, h);

        ctx.strokeStyle = "rgba(180, 130, 0, 0.85)";
        ctx.lineWidth   = Math.max(1.5, canvas.width / 800);
        ctx.strokeRect(x, y, w, h);

        ctx.font        = `bold ${fontSize}px sans-serif`;
        const padX      = 6, padY = 4;
        const textW     = ctx.measureText(keyword).width;
        const pillW     = textW + padX * 2;
        const pillH     = fontSize + padY * 2;
        const pillX     = Math.max(0, Math.min(x, canvas.width - pillW));
        const pillY     = Math.max(0, y - pillH - 2);

        ctx.fillStyle = "rgba(255, 200, 0, 0.95)";
        if (ctx.roundRect) {
          ctx.beginPath();
          ctx.roundRect(pillX, pillY, pillW, pillH, 4);
          ctx.fill();
        } else {
          ctx.fillRect(pillX, pillY, pillW, pillH);
        }

        ctx.fillStyle = "#3d2800";
        ctx.fillText(keyword, pillX + padX, pillY + fontSize + padY - 2);
      });

      const legendTxt = `${wordBoxes.length} keyword match${wordBoxes.length !== 1 ? "es" : ""}`;
      ctx.font        = `${Math.max(11, fontSize - 2)}px sans-serif`;
      const legW      = ctx.measureText(legendTxt).width + 16;
      const legH      = fontSize + 10;
      const legX      = canvas.width - legW - 10;
      const legY      = 10;

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

  const handleDownload = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link    = document.createElement("a");
    const base    = (filename || "image").replace(/\.[^/.]+$/, "");
    link.download = `${base}_keywords_highlighted.png`;
    link.href     = canvas.toDataURL("image/png");
    link.click();
  };

  if (!originalB64) return null;

  return (
    <div className="highlight-container">
      <div className="highlight-header">
        <span className="highlight-title">
          🟡 Original Image — Keyword Highlights (exact Tesseract bounding boxes)
        </span>
        <button className="download-btn" onClick={handleDownload}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none"
            stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
            <polyline points="7 10 12 15 17 10"/>
            <line x1="12" y1="15" x2="12" y2="3"/>
          </svg>
          Download Highlighted
        </button>
      </div>
      <canvas ref={canvasRef} className="highlight-canvas" />
      {wordBoxes && wordBoxes.length > 0 && (
        <p className="highlight-note">
          ✦ Highlights use exact Tesseract OCR word coordinates — perfectly sized bounding boxes
        </p>
      )}
    </div>
  );
}


function ImageComparison({ preprocessedB64 }) {
  if (!preprocessedB64) return null;
  return (
    <div className="img-comparison">
      <div className="img-box">
        <span className="img-label preprocessed">CV Preprocessed Output</span>
        <img src={`data:image/png;base64,${preprocessedB64}`} alt="CV Preprocessed" className="compare-img" />
        <span className="img-caption">
          upscale → deskew → gaussian denoise → CLAHE → unsharp mask → adaptive threshold → morph close
        </span>
      </div>
    </div>
  );
}


function AnalysisBadges({ topic, confidence, readability, word_count }) {
  if (!topic) return null;

  const confPct  = confidence ? Math.round(confidence * 100) : 0;
  const readColor =
    (readability?.score ?? 0) >= 60 ? "var(--accent3)" :
    (readability?.score ?? 0) >= 40 ? "#f59e0b" : "#f87171";

  return (
    <div className="analysis-grid">
      <div className="analysis-card">
        <span className="ac-label">Topic (Zero-shot DL)</span>
        <span className="ac-value topic">{topic}</span>
        <span className="ac-sub">BART-MNLI · {confPct}% confident</span>
      </div>
      <div className="analysis-card">
        <span className="ac-label">Readability (OCR text)</span>
        <span className="ac-value" style={{ color: readColor }}>{readability?.score ?? "—"}</span>
        <span className="ac-sub">{readability?.level ?? ""}</span>
      </div>
      <div className="analysis-card">
        <span className="ac-label">Model Confidence</span>
        <div className="conf-bar-wrap">
          <div className="conf-bar" style={{ width: `${confPct}%` }} />
        </div>
        <span className="ac-sub">{confPct}%</span>
      </div>
      <div className="analysis-card">
        <span className="ac-label">Word Count</span>
        <span className="ac-value">{word_count ?? "—"}</span>
        <span className="ac-sub">from OCR output</span>
      </div>
    </div>
  );
}


function KeywordCloud({ keywords }) {
  if (!keywords || keywords.length === 0) return null;
  return (
    <div className="keyword-section">
      <span className="section-label">🔑 TF-IDF Keywords (ranked by score)</span>
      <div className="keyword-cloud">
        {keywords.map((kw, i) => (
          <span key={i} className="kw-pill" style={{ fontSize: `${Math.max(0.72, 0.9 - i * 0.02)}rem` }}>
            {kw}
          </span>
        ))}
      </div>
    </div>
  );
}


function ResultCard({ result, filename }) {
  const [ocrExpanded, setOcrExpanded] = useState(false);

  return (
    <div className="result-card">
      <KeywordHighlighter
        originalB64={result.original_image_b64}
        originalMime={result.original_image_mime}
        wordBoxes={result.word_boxes}
        filename={filename}
      />
      <h2 className="result-heading">{result.heading}</h2>
      <AnalysisBadges
        topic={result.topic}
        confidence={result.confidence}
        readability={result.readability}
        word_count={result.word_count}
      />
      <KeywordCloud keywords={result.keywords} />
      <div className="result-section">
        <span className="section-label">📝 Summary</span>
        <p className="result-summary">{result.summary}</p>
      </div>
      {result.visual_elements && result.visual_elements !== "None" && (
        <div className="result-section">
          <span className="section-label">🖼️ Visual Elements</span>
          <p className="result-visual">{result.visual_elements}</p>
        </div>
      )}
      {result.preprocessed_image_b64 && (
        <div className="result-section">
          <span className="section-label">🔬 CV Preprocessed Image</span>
          <ImageComparison preprocessedB64={result.preprocessed_image_b64} />
        </div>
      )}
      {result.extracted_text && (
        <div className="result-section">
          <button className="expand-btn" onClick={() => setOcrExpanded(!ocrExpanded)}>
            {ocrExpanded ? "▲ Hide" : "▼ Show"} Raw OCR Text
          </button>
          {ocrExpanded && <pre className="extracted-text">{result.extracted_text}</pre>}
        </div>
      )}
    </div>
  );
}


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


export default function App() {
  const [mode,       setMode]       = useState("image");
  const [file,       setFile]       = useState(null);
  const [preview,    setPreview]    = useState(null);
  const [loading,    setLoading]    = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("");
  const [result,     setResult]     = useState(null);
  const [error,      setError]      = useState(null);

  const handleFile = (f) => {
    setFile(f);
    setResult(null);
    setError(null);
    setPreview(f.type.startsWith("image/") ? URL.createObjectURL(f) : null);
  };

  const handleAnalyse = async () => {
    if (!file) return;
    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("file", file);

    const endpoint = mode === "image" ? "/analyse/image" : "/analyse/pdf";

    const msgs = mode === "image"
      ? [
          "Reading original image...",
          "Running CV preprocessing pipeline...",
          "Deskewing & denoising...",
          "CLAHE contrast enhancement...",
          "Sending to VLM (OCR + captioning)...",
          "Running TF-IDF keyword extraction...",
          "Zero-shot topic classification (BART-MNLI)...",
          "Computing Flesch readability score...",
          "Building your study notes...",
        ]
      : [
          "Rasterising PDF pages (PyMuPDF)...",
          "Running CV pipeline on each page...",
          "OCR analysis...",
          "Generating document summary...",
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

  const reset = () => {
    setFile(null);
    setPreview(null);
    setResult(null);
    setError(null);
  };

  return (
    <div className="app">
      <div className="bg-orb orb1" />
      <div className="bg-orb orb2" />
      <div className="bg-orb orb3" />

      <header className="header">
        <div className="logo">
          <span className="logo-icon">◈</span>
          <span className="logo-text">SnapWise</span>
        </div>
        <p className="tagline">
          
        </p>
      </header>

      <main className="main">
        <div className="mode-toggle">
          <button className={`mode-btn ${mode === "image" ? "active" : ""}`}
            onClick={() => { setMode("image"); reset(); }}>
            🖼️ Screenshot
          </button>
          <button className={`mode-btn ${mode === "pdf" ? "active" : ""}`}
            onClick={() => { setMode("pdf"); reset(); }}>
            📄 PDF Document
          </button>
        </div>

        {!result && (
          <section className="upload-section">
            <FileDropzone
              onFile={handleFile}
              accept={mode === "image" ? "image/*" : "application/pdf"}
              label={mode === "image" ? "Upload a Screenshot" : "Upload a PDF"}
              icon={mode === "image" ? "🖼️" : "📄"}
              disabled={loading}
            />
            {file && (
              <div className="file-info">
                <span className="file-name">✓ {file.name}</span>
                <span className="file-size">{(file.size / 1024).toFixed(1)} KB</span>
              </div>
            )}
            {preview && (
              <div className="preview-container">
                <img src={preview} alt="Upload preview" className="preview-img" />
              </div>
            )}
            {file && !loading && (
              <button className="analyse-btn" onClick={handleAnalyse}>
                <span>Analyse with AI</span>
                <span className="btn-arrow">→</span>
              </button>
            )}
          </section>
        )}

        {loading && <LoadingPulse message={loadingMsg} />}

        {error && (
          <div className="error-box">
            <span>⚠️ {error}</span>
            <button onClick={reset} className="retry-btn">Try Again</button>
          </div>
        )}

        {result && !loading && (
          <section className="results-section">
            <div className="results-header">
              <h1 className="results-title">Analysis Complete</h1>
              <button className="new-btn" onClick={reset}>+ New Analysis</button>
            </div>

            {result.type === "image" && (
              <ResultCard result={result.result} filename={file?.name} />
            )}

            {result.type === "pdf" && (
                <>
                  <div className="pdf-meta">
                    📑 {result.total_pages} pages total · {result.analysed_pages} analysed
                  </div>
                  <div className="result-card">
                    {result.pages && result.pages.map((page) => (
                      <KeywordHighlighter
                        key={page.page}
                        originalB64={page.original_image_b64}
                        originalMime="image/png"
                        wordBoxes={page.word_boxes}
                        filename={`${file?.name || "page"}_p${page.page}`}
                      />
                    ))}
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
                </>
              )}
          </section>
        )}
      </main>

      <footer className="footer">
        <p>
          <br />SnapWise 
        </p>
      </footer>
    </div>
  );
}
