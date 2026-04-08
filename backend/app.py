import base64
import io
import json
import os
import re

from flask import Flask, request, jsonify
from flask_cors import CORS

import cv2
import numpy as np
from PIL import Image
import fitz
import easyocr
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline as hf_pipeline
from openai import OpenAI


# ═══════════════════════════════════════════════════════
# APP INIT
# ═══════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)

HF_TOKEN    = os.environ.get("HF_TOKEN", "YOUR_HF_TOKEN_HERE")
VLM_MODEL = "CohereLabs/aya-vision-32b:cohere"
VLM_MAX_TOK = 1024
vlm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=HF_TOKEN,
)

print("Loading zero-shot classifier...")
try:
    zero_shot    = hf_pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=-1)
    ZS_AVAILABLE = True
    print("Zero-shot classifier loaded ✓")
except Exception as e:
    print(f"Warning: zero-shot unavailable ({e})")
    zero_shot    = None
    ZS_AVAILABLE = False

print("Loading EasyOCR...")
try:
    ocr_reader    = easyocr.Reader(['en'], gpu=False, verbose=False)
    OCR_AVAILABLE = True
    print("EasyOCR loaded ✓")
except Exception as e:
    print(f"Warning: EasyOCR unavailable ({e})")
    ocr_reader    = None
    OCR_AVAILABLE = False

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_PDF_TYPE    = "application/pdf"

TOPIC_LABELS = [
    "Mathematics", "Physics", "Chemistry", "Biology", "Computer Science",
    "History", "Geography", "Economics", "Literature", "Engineering",
    "Data Science", "Machine Learning", "Programming", "General Knowledge",
]

VLM_PROMPT = (
    "You are an intelligent academic assistant helping students study.\n\n"
    "Carefully analyse the provided image and do ALL of the following:\n"
    "  1. Extract ALL visible text exactly as it appears (OCR).\n"
    "  2. Describe any diagrams, charts, tables, equations, or visual elements.\n"
    "  3. Write a concise HEADING (max 10 words) that captures the core topic.\n"
    "  4. Write a SUMMARY (150-250 words) explaining the content for a student.\n\n"
    "Respond ONLY in this exact JSON format — no markdown, no extra keys:\n"
    '{"heading":"...","summary":"...","extracted_text":"...","visual_elements":"..."}'
)


# ═══════════════════════════════════════════════════════
# CV PREPROCESSING
# ═══════════════════════════════════════════════════════

def preprocess_image(image_bytes: bytes) -> tuple[bytes, str]:
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    w, h    = pil_img.size

    if min(w, h) < 640:
        scale   = 640 / min(w, h)
        pil_img = pil_img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    img_rgb = np.array(pil_img)
    gray    = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray    = _deskew(gray)

    denoised  = cv2.GaussianBlur(gray, (3, 3), 0)
    clahe     = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced  = clahe.apply(denoised)
    blur      = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)

    binary = cv2.adaptiveThreshold(
        sharpened, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31, C=10,
    )

    kernel     = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed     = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    closed_3ch = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    blended    = cv2.addWeighted(closed_3ch, 0.6, img_rgb, 0.4, 0)

    result_pil = Image.fromarray(blended)
    w, h       = result_pil.size
    if max(w, h) > 1024:
        scale      = 1024 / max(w, h)
        result_pil = result_pil.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    result_pil.save(buf, format="JPEG", quality=85)
    processed_bytes = buf.getvalue()
    preview_b64     = base64.standard_b64encode(processed_bytes).decode("utf-8")

    return processed_bytes, preview_b64


def _deskew(gray: np.ndarray) -> np.ndarray:
    edges  = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines  = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    if lines is None:
        return gray

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 != x1:
            angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
            if -15.0 < angle < 15.0:
                angles.append(angle)

    if not angles or abs(float(np.median(angles))) < 0.5:
        return gray

    h, w = gray.shape
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), float(np.median(angles)), 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


# ═══════════════════════════════════════════════════════
# VLM ANALYSIS
# ═══════════════════════════════════════════════════════

def vlm_analyse(processed_bytes: bytes) -> dict:
    b64      = base64.standard_b64encode(processed_bytes).decode("utf-8")
    data_url = f"data:image/jpeg;base64,{b64}"

    completion = vlm_client.chat.completions.create(
        model=VLM_MODEL,
        max_tokens=VLM_MAX_TOK,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url}},
                {"type": "text",      "text": VLM_PROMPT},
            ],
        }],
    )

    raw = completion.choices[0].message.content.strip()
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


# ═══════════════════════════════════════════════════════
# ML / DL POST-PROCESSING
# ═══════════════════════════════════════════════════════

def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    if not text or len(text.strip()) < 20:
        return []

    sentences = [s.strip() for s in re.split(r'[.!?\n]+', text) if len(s.strip()) > 5]

    if len(sentences) < 2:
        from collections import Counter
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        stops = {"this","that","with","from","have","been","they","were","their","will","when","what","also","some","more","into"}
        return [w for w, _ in Counter(words).most_common(top_n + 20) if w not in stops][:top_n]

    vectoriser = TfidfVectorizer(ngram_range=(1, 2), stop_words="english", max_df=0.85, min_df=1, max_features=500)
    try:
        tfidf_matrix  = vectoriser.fit_transform(sentences)
        scores        = tfidf_matrix.sum(axis=0).A1
        feature_names = vectoriser.get_feature_names_out()
        return [feature_names[i] for i in scores.argsort()[::-1][:top_n]]
    except ValueError:
        return []


def classify_topic(text: str) -> tuple[str, float]:
    if not ZS_AVAILABLE or not text or len(text.strip()) < 20:
        return "General Knowledge", 0.0
    try:
        result = zero_shot(text[:512], candidate_labels=TOPIC_LABELS, multi_label=False)
        return result["labels"][0], round(result["scores"][0], 3)
    except Exception as e:
        print(f"Zero-shot error: {e}")
        return "General Knowledge", 0.0


def flesch_reading_ease(text: str) -> dict:
    if not text or len(text.strip()) < 30:
        return {"score": 0.0, "level": "N/A", "avg_sentence_len": 0, "avg_syllables_per_word": 0}

    sentences   = [s for s in re.split(r'[.!?]+\s*', text.strip()) if len(s.split()) > 0]
    n_sentences = max(len(sentences), 1)
    words       = re.findall(r'[a-zA-Z]+', text)
    n_words     = max(len(words), 1)

    def count_syllables(word):
        word  = word.lower()
        count = len(re.findall(r'[aeiou]+', word))
        if word.endswith('e') and len(word) > 2:
            count -= 1
        return max(count, 1)

    n_syllables      = sum(count_syllables(w) for w in words)
    avg_sent_len     = n_words / n_sentences
    avg_syll_per_word = n_syllables / n_words
    score            = max(0.0, min(100.0, round(206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syll_per_word), 1)))

    level = ("Very Easy" if score >= 90 else "Easy" if score >= 70 else "Standard"
             if score >= 60 else "Fairly Difficult" if score >= 50 else "Difficult"
             if score >= 30 else "Very Confusing (Academic)")

    return {
        "score": score, "level": level,
        "avg_sentence_len": round(avg_sent_len, 1),
        "avg_syllables_per_word": round(avg_syll_per_word, 2),
    }


def get_keyword_bounding_boxes(image_bytes: bytes, keywords: list[str]) -> list[dict]:
    if not OCR_AVAILABLE or not keywords:
        return []

    try:
        img_np      = np.array(Image.open(io.BytesIO(image_bytes)).convert("RGB"))
        raw_results = ocr_reader.readtext(img_np, detail=1, paragraph=False, width_ths=0.7)

        word_records = []
        for (polygon, text, conf) in raw_results:
            if conf < 0.3 or not text.strip():
                continue
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            word_records.append({
                "text": text.strip().lower(),
                "x": int(min(xs)), "y": int(min(ys)),
                "w": int(max(xs) - min(xs)), "h": int(max(ys) - min(ys)),
            })

        result_boxes = []
        used_indices = set()

        for kw in keywords:
            kw_lower  = kw.lower().strip()
            kw_tokens = kw_lower.split()
            n_tokens  = len(kw_tokens)

            if n_tokens == 1:
                for idx, rec in enumerate(word_records):
                    if idx not in used_indices and (rec["text"] == kw_lower or rec["text"].startswith(kw_lower)):
                        result_boxes.append({"keyword": kw, "x": rec["x"], "y": rec["y"], "w": rec["w"], "h": rec["h"]})
                        used_indices.add(idx)
            else:
                for i in range(len(word_records) - n_tokens + 1):
                    if any((i + j) in used_indices for j in range(n_tokens)):
                        continue
                    match, window = True, []
                    for j in range(n_tokens):
                        rec, tok = word_records[i + j], kw_tokens[j]
                        if not (rec["text"] == tok or rec["text"].startswith(tok)):
                            match = False; break
                        window.append(rec)
                    if not match:
                        continue
                    if max(r["y"] for r in window) - min(r["y"] for r in window) > 20:
                        continue
                    result_boxes.append({
                        "keyword": kw,
                        "x": min(r["x"] for r in window),
                        "y": min(r["y"] for r in window),
                        "w": max(r["x"] + r["w"] for r in window) - min(r["x"] for r in window),
                        "h": max(r["y"] + r["h"] for r in window) - min(r["y"] for r in window),
                    })
                    for j in range(n_tokens):
                        used_indices.add(i + j)

        return result_boxes

    except Exception as e:
        print(f"EasyOCR error: {e}")
        return []


def run_ml_dl_analysis(vlm_result: dict) -> dict:
    extracted_text       = vlm_result.get("extracted_text", "")
    summary              = vlm_result.get("summary", "")
    heading              = vlm_result.get("heading", "")
    classification_input = f"{heading}. {summary}"

    keywords          = extract_keywords(extracted_text, top_n=10)
    topic, confidence = classify_topic(classification_input)
    readability       = flesch_reading_ease(extracted_text or summary)
    word_count        = len(re.findall(r'\b\w+\b', extracted_text))

    return {
        "keywords": keywords, "topic": topic,
        "confidence": confidence, "readability": readability,
        "word_count": word_count,
    }


# ═══════════════════════════════════════════════════════
# PDF UTILITIES
# ═══════════════════════════════════════════════════════

def pdf_page_to_png_bytes(page: fitz.Page) -> bytes:
    return page.get_pixmap(matrix=fitz.Matrix(2, 2)).tobytes("png")


# ═══════════════════════════════════════════════════════
# FLASK ROUTES
# ═══════════════════════════════════════════════════════

@app.route("/api/analyse/image", methods=["POST"])
def analyse_image():
    if "file" not in request.files:
        return jsonify({"error": "No file field in request."}), 400

    f = request.files["file"]
    if f.content_type not in ALLOWED_IMAGE_TYPES:
        return jsonify({"error": f"Unsupported type '{f.content_type}'."}), 400

    image_bytes = f.read()

    try:
        original_b64  = base64.standard_b64encode(image_bytes).decode("utf-8")
        original_mime = f.content_type

        processed_bytes, preprocessed_b64 = preprocess_image(image_bytes)
        vlm_result  = vlm_analyse(processed_bytes)
        ml_result   = run_ml_dl_analysis(vlm_result)
        word_boxes  = get_keyword_bounding_boxes(image_bytes, ml_result.get("keywords", []))

        return jsonify({
            "success": True,
            "type": "image",
            "result": {
                **vlm_result, **ml_result,
                "word_boxes":             word_boxes,
                "preprocessed_image_b64": preprocessed_b64,
                "original_image_b64":     original_b64,
                "original_image_mime":    original_mime,
            },
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"VLM returned malformed JSON: {e}"}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyse/pdf", methods=["POST"])
def analyse_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file field in request."}), 400

    f = request.files["file"]
    if f.content_type != ALLOWED_PDF_TYPE:
        return jsonify({"error": f"Expected PDF, got '{f.content_type}'."}), 400

    pdf_bytes = f.read()

    try:
        doc           = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages   = len(doc)
        max_pages     = min(total_pages, 10)
        pages_results = []

        for page_num in range(max_pages):
            raw_png           = pdf_page_to_png_bytes(doc[page_num])
            page_original_b64 = base64.standard_b64encode(raw_png).decode("utf-8")
            processed_bytes, _ = preprocess_image(raw_png)
            vlm_result        = vlm_analyse(processed_bytes)
            page_keywords     = extract_keywords(vlm_result.get("extracted_text", ""), top_n=10)
            page_word_boxes   = get_keyword_bounding_boxes(raw_png, page_keywords)

            pages_results.append({
                "page":                page_num + 1,
                "original_image_b64":  page_original_b64,
                "original_image_mime": "image/png",
                "word_boxes":          page_word_boxes,
                **vlm_result,
            })

        combined_text   = "\n\n".join(p.get("extracted_text", "") for p in pages_results)[:6000]
        overall_prompt  = (
            f"This is the full text of a multi-page academic document:\n\n{combined_text}\n\n"
            "Write a document-level heading (max 10 words) and a comprehensive summary "
            "(200-300 words) that a student can use as study notes.\n"
            "Respond ONLY in this exact JSON with no markdown:\n"
            '{"heading":"...","summary":"..."}'
        )

        overall_resp = vlm_client.chat.completions.create(
            model=VLM_MODEL, max_tokens=512,
            messages=[{"role": "user", "content": overall_prompt}],
        )
        raw_overall = overall_resp.choices[0].message.content.strip()
        if raw_overall.startswith("```"):
            parts       = raw_overall.split("```")
            raw_overall = parts[1][4:].strip() if parts[1].startswith("json") else parts[1].strip()
        overall_vlm = json.loads(raw_overall)
        overall_ml  = run_ml_dl_analysis({
            "extracted_text": combined_text,
            "summary":        overall_vlm.get("summary", ""),
            "heading":        overall_vlm.get("heading", ""),
        })

        return jsonify({
            "success": True, "type": "pdf",
            "total_pages": total_pages, "analysed_pages": max_pages,
            "overall": {**overall_vlm, **overall_ml},
            "pages":   pages_results,
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"VLM returned malformed JSON: {e}"}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status":    "ok",
        "vlm_model": VLM_MODEL,
        "zero_shot": ZS_AVAILABLE,
        "easyocr":   OCR_AVAILABLE,
        "opencv":    cv2.__version__,
    })


if __name__ == "__main__":
    print("=" * 55)
    print("  SnapWise Backend")
    print("=" * 55)
    print(f"  Model    : {VLM_MODEL}")
    print(f"  HF Token : {'SET ✓' if HF_TOKEN != 'YOUR_HF_TOKEN_HERE' else 'NOT SET ✗'}")
    print(f"  Zero-shot: {'✓' if ZS_AVAILABLE else '✗'}")
    print(f"  EasyOCR  : {'✓' if OCR_AVAILABLE else '✗'}")
    print(f"  OpenCV   : {cv2.__version__}")
    print("=" * 55)
    app.run(debug=True, port=5000)
