"""
╔══════════════════════════════════════════════════════════════════════════════╗
║         SnapWise — Screenshot & PDF Analyser (Interview-Grade)              ║
║                                                                              ║
║  TECH STACK:                                                                 ║
║  • Flask        — lightweight REST API server                                ║
║  • OpenCV       — classical Computer Vision preprocessing pipeline           ║
║  • Pillow       — image I/O and format conversion                            ║
║  • PyMuPDF      — PDF page rasterisation                                     ║
║  • scikit-learn — ML: TF-IDF keyword extraction + topic clustering           ║
║  • transformers — DL: zero-shot text classification (BART-MNLI)              ║
║  • OpenAI SDK   → HuggingFace router — Vision-Language Model (Qwen2.5-VL)   ║
║                                                                              ║
║  PIPELINE OVERVIEW:                                                          ║
║  Raw Image/PDF                                                               ║
║      │                                                                       ║
║      ▼                                                                       ║
║  [1] CV PREPROCESSING  (OpenCV)                                              ║
║      upscale → deskew → denoise → CLAHE → sharpen → binarise → morph        ║
║      │                                                                       ║
║      ▼                                                                       ║
║  [2] VLM ANALYSIS  (Qwen2.5-VL via HF router)                               ║
║      OCR + image captioning → heading + summary JSON                         ║
║      │                                                                       ║
║      ▼                                                                       ║
║  [3] ML/DL POST-PROCESSING  (scikit-learn + transformers)                    ║
║      TF-IDF keywords + zero-shot topic classification + readability score    ║
║      │                                                                       ║
║      ▼                                                                       ║
║  [4] RESPONSE                                                                ║
║      JSON  { heading, summary, extracted_text, visual_elements,              ║
║              keywords, topic, confidence, readability,                       ║
║              preprocessed_image_b64 }                                        ║
╚══════════════════════════════════════════════════════════════════════════════╝

AUTHOR  : Shlok — built for internship/interview portfolio
DATE    : 2025
"""

# ─────────────────────────────────────────────────────────────────────────────
# STANDARD LIBRARY IMPORTS
# ─────────────────────────────────────────────────────────────────────────────
import base64          # encode image bytes → base64 string for JSON transport
import io              # in-memory byte streams (no temp files needed)
import json            # parse / serialise JSON between Python and clients
import os              # read environment variables (HF_TOKEN)
import re              # regex for text cleaning before ML processing
import math            # syllable counting for readability (Flesch-Kincaid)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY: WEB FRAMEWORK
# ─────────────────────────────────────────────────────────────────────────────
from flask import Flask, request, jsonify   # micro web-framework for REST API
from flask_cors import CORS                  # allow React frontend (port 3000) to call Flask (port 5000)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY: COMPUTER VISION  (OpenCV + NumPy + Pillow)
# ─────────────────────────────────────────────────────────────────────────────
import cv2             # OpenCV — core CV operations (blur, threshold, morph, Hough)
import numpy as np     # NumPy  — OpenCV works on numpy arrays
from PIL import Image  # Pillow — image I/O, format conversion, colour modes

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY: PDF PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
import fitz            # PyMuPDF — renders PDF pages to pixel bitmaps at any DPI

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY: OCR WITH BOUNDING BOXES  (easyocr)
# ─────────────────────────────────────────────────────────────────────────────
import easyocr
# EasyOCR is a pure-Python deep learning OCR library — NO external binary
# installation needed (unlike Tesseract which requires a system install).
#
# How EasyOCR works:
#   • Uses a CRAFT text detector (convolutional neural net) to find text regions
#   • Then runs a sequence-to-sequence recognition model (CRNN) on each region
#   • Returns: [[bbox_polygon, text, confidence], ...]
#     where bbox_polygon = [[x1,y1],[x2,y1],[x2,y2],[x1,y2]] (4 corner points)
#
# We convert the 4-corner polygon → axis-aligned (x, y, w, h) bounding box,
# then match each detected word against TF-IDF keywords to find exact locations.
#
# EasyOCR reader is initialised ONCE at startup and reused — model loading
# takes ~5 seconds; per-call inference is fast (~0.5-2s depending on image size).
# (numpy already imported above)

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY: ML — CLASSICAL  (scikit-learn)
# ─────────────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
# TF-IDF = Term Frequency × Inverse Document Frequency
# TF(t)  = (occurrences of term t in doc) / (total terms in doc)
# IDF(t) = log(total docs / docs containing t)  — penalises common words
# TF-IDF score ↑  →  word is important to THIS doc but rare across corpus
# We use single-document mode: treats each sentence as a "document"
# to surface the most discriminative terms in the extracted text.

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY: DL — ZERO-SHOT CLASSIFICATION  (HuggingFace transformers)
# ─────────────────────────────────────────────────────────────────────────────
from transformers import pipeline as hf_pipeline
# We load facebook/bart-large-mnli — a BART model fine-tuned on Multi-NLI
# MNLI = Multi-Genre Natural Language Inference dataset
# Zero-shot classification works via NLI:
#   premise   = the text we want to classify
#   hypothesis = "This text is about {label}"
#   model outputs: entailment / neutral / contradiction probability
#   entailment score = confidence that label fits → we use this as our score
# No training data needed — the model generalises to unseen label sets.

# ─────────────────────────────────────────────────────────────────────────────
# THIRD-PARTY: VLM API  (OpenAI SDK → HuggingFace router)
# ─────────────────────────────────────────────────────────────────────────────
from openai import OpenAI
# HuggingFace's router.huggingface.co exposes an OpenAI-compatible /v1 endpoint
# so we can use the standard openai Python SDK — just swap base_url.
# The model we call: Qwen/Qwen2.5-VL-7B-Instruct
#   • 7B parameter vision-language model by Alibaba Qwen team
#   • Trained on both image-text pairs and pure text
#   • Supports: OCR, visual QA, image captioning, document understanding
#   • Input: base64-encoded image + text prompt via chat/completions API


# ═════════════════════════════════════════════════════════════════════════════
# APP INITIALISATION
# ═════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
CORS(app)  # allow cross-origin requests from the React dev server

# ── API credentials ──────────────────────────────────────────────────────────
# os.environ.get("HF_TOKEN", fallback) reads the env var;
# if not set, uses the hardcoded fallback so local dev works without export.
HF_TOKEN = os.environ.get("HF_TOKEN","UR TOKEN")

# ── Vision-Language Model config ─────────────────────────────────────────────
VLM_MODEL   = "Qwen/Qwen2.5-VL-7B-Instruct"
VLM_MAX_TOK = 1024   # max output tokens — enough for OCR + summary JSON

# ── OpenAI SDK pointed at HuggingFace router ─────────────────────────────────
vlm_client = OpenAI(
    base_url="https://router.huggingface.co/v1",  # HF's new unified router
    api_key=HF_TOKEN,
)

# ── Zero-shot classifier (loaded once at startup, cached in memory) ───────────
# Loading a transformer model takes ~2-3 seconds; doing it per-request is slow.
# We load it once here so all requests share the same in-memory model instance.
print("Loading zero-shot classifier (facebook/bart-large-mnli)...")
try:
    zero_shot = hf_pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        # device=-1 means CPU — change to device=0 for GPU acceleration
        device=-1,
    )
    ZS_AVAILABLE = True
    print("Zero-shot classifier loaded ✓")
except Exception as e:
    # Graceful degradation: if transformers/model not available, skip DL step
    print(f"Warning: zero-shot classifier unavailable ({e}). Skipping DL step.")
    zero_shot = None
    ZS_AVAILABLE = False

# ── EasyOCR reader — loaded once at startup, shared across all requests ───────
# gpu=False forces CPU inference — works on any machine without CUDA.
# ['en'] = English language model. Add more langs e.g. ['en','hi'] for Hindi.
# First run downloads ~100MB model weights into ~/.EasyOCR/model/
print("Loading EasyOCR reader...")
try:
    ocr_reader   = easyocr.Reader(['en'], gpu=False, verbose=False)
    OCR_AVAILABLE = True
    print("EasyOCR reader loaded ✓")
except Exception as e:
    print(f"Warning: EasyOCR unavailable ({e}). Keyword highlighting disabled.")
    ocr_reader   = None
    OCR_AVAILABLE = False

# ── File-type allowlists ──────────────────────────────────────────────────────
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/gif", "image/webp"}
ALLOWED_PDF_TYPE    = "application/pdf"

# ── Academic topic labels for zero-shot classification ───────────────────────
# These are the candidate labels the BART-MNLI model will score against.
# Add or modify labels to match your domain (e.g. add "law", "medicine").
TOPIC_LABELS = [
    "Mathematics",
    "Physics",
    "Chemistry",
    "Biology",
    "Computer Science",
    "History",
    "Geography",
    "Economics",
    "Literature",
    "Engineering",
    "Data Science",
    "Machine Learning",
    "Programming",
    "General Knowledge",
]

# ── Prompt sent to the VLM alongside each image ───────────────────────────────
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


# ═════════════════════════════════════════════════════════════════════════════
#  BLOCK 1 — COMPUTER VISION PREPROCESSING PIPELINE
#  Purpose: clean and normalise the image so the VLM can read it accurately.
#  Input  : raw image bytes (any format Pillow can open)
#  Output : preprocessed PNG bytes + base64 string for frontend preview
# ═════════════════════════════════════════════════════════════════════════════

def preprocess_image(image_bytes: bytes) -> tuple[bytes, str]:
    """
    Run the 7-stage CV preprocessing pipeline on a raw image.

    Returns
    -------
    preprocessed_bytes : bytes
        PNG bytes of the cleaned image — sent to the VLM.
    preview_b64 : str
        Base64-encoded PNG — sent to the frontend for side-by-side preview.

    Pipeline stages
    ---------------
    1. Decode & upscale   — ensure minimum 640 px on shortest side
    2. Deskew             — Hough line transform to correct rotation
    3. Gaussian denoise   — 3×3 kernel smooths JPEG/compression noise
    4. CLAHE              — local contrast enhancement (adaptive histogram EQ)
    5. Unsharp mask       — sharpens text edges via high-frequency boost
    6. Adaptive threshold — binarise for clean black-on-white text
    7. Morphological close— reconnect broken character strokes
    8. Colour blend       — merge with original so diagrams stay visible
    """

    # ── Stage 1: Decode & Upscale ─────────────────────────────────────────────
    # PIL opens any image format (JPEG, PNG, WebP, GIF).
    # .convert("RGB") normalises to 3-channel — removes alpha, handles palette images.
    pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    w, h = pil_img.size
    min_side = min(w, h)

    # VLMs process images at a fixed internal resolution (~448 or ~896 px).
    # If the input is smaller, upscaling avoids lossy downsizing that blurs text.
    if min_side < 640:
        scale   = 640 / min_side
        new_w   = int(w * scale)
        new_h   = int(h * scale)
        # LANCZOS (Lanczos resampling) = high-quality sinc-based filter
        # Preserves fine detail better than BILINEAR or BICUBIC for upscaling.
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

    # Convert PIL → NumPy array for OpenCV (which works on np.ndarray)
    # Shape: (H, W, 3) with values 0-255 in RGB order
    img_rgb = np.array(pil_img)

    # Convert RGB → Grayscale for all subsequent CV operations
    # Formula: gray = 0.299·R + 0.587·G + 0.114·B  (BT.601 luma coefficients)
    # Grayscale removes colour noise and reduces computation.
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    # ── Stage 2: Deskew via Hough Line Transform ──────────────────────────────
    # Screenshots taken at an angle or photos of pages are often slightly rotated.
    # Even 2° skew significantly degrades OCR accuracy.
    gray = _deskew(gray)

    # ── Stage 3: Gaussian Denoising ───────────────────────────────────────────
    # Gaussian blur with kernel size 3×3:
    #   • Each pixel becomes a weighted average of its 3×3 neighbourhood
    #   • Weights follow a 2D Gaussian bell curve (centre pixel weighs most)
    # sigma=0 → OpenCV auto-calculates σ from kernel size: σ ≈ 0.3*(k-1)/2 + 0.8
    # Effect: smooths high-frequency noise (JPEG artefacts, sensor noise)
    #         while preserving low-frequency structure (text strokes, edges).
    # We use a SMALL kernel (3×3) intentionally — just enough to remove noise
    # without blurring the thin strokes of small text.
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)

    # ── Stage 4: CLAHE — Contrast Limited Adaptive Histogram Equalisation ─────
    # Problem: screenshots often have uneven illumination (dark corners, bright
    #          backgrounds, mixed dark/light text regions).
    # Global histogram equalisation (cv2.equalizeHist) enhances contrast globally
    # but over-amplifies noise in already-bright regions and loses detail.
    #
    # CLAHE solution:
    #   • Divides image into tileGridSize×tileGridSize non-overlapping tiles (8×8)
    #   • Applies histogram equalisation independently to each tile
    #   • clipLimit=2.0 caps the histogram redistribution — prevents noise amplification
    #   • Bilinear interpolation stitches tile borders smoothly
    # Result: dark text on dark background becomes readable,
    #         bright areas are not blown out.
    clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(denoised)

    # ── Stage 5: Unsharp Masking (Sharpening) ─────────────────────────────────
    # Idea: sharpen = original + α × (original - blurred)
    #       equivalently: sharpen = (1+α)×original - α×blurred
    #
    # Here: sharpen = 1.5 × enhanced - 0.5 × blur(enhanced, σ=3)
    #
    # Why σ=3 for the blur? A larger σ captures broader detail loss;
    # subtracting it amplifies medium-frequency edges (text strokes ~3-8px wide).
    # σ=1 would only sharpen noise; σ>5 would over-sharpen and create haloes.
    blur      = cv2.GaussianBlur(enhanced, (0, 0), sigmaX=3)
    sharpened = cv2.addWeighted(enhanced, 1.5, blur, -0.5, 0)
    # addWeighted(src1, α, src2, β, γ) = α·src1 + β·src2 + γ
    # Values are clipped to [0, 255] automatically.

    # ── Stage 6: Adaptive Thresholding ────────────────────────────────────────
    # Binarisation: convert grayscale → pure black (0) / white (255).
    # Purpose: eliminates background gradients, shadows, and colour variation
    #          so text is stark black on white — ideal for OCR.
    #
    # Why ADAPTIVE not global Otsu?
    #   Otsu picks ONE global threshold T: pixel > T → white, else black.
    #   Works only if illumination is uniform across the whole image.
    #   Screenshots often have mixed light/dark regions → Otsu fails.
    #
    # ADAPTIVE_THRESH_GAUSSIAN_C:
    #   For each pixel p, compute local threshold T(p):
    #     T(p) = weighted_mean(neighbourhood of blockSize×blockSize centred on p) - C
    #   where weights are a Gaussian (centre pixel has highest weight).
    #   blockSize=31 → 31×31 pixel neighbourhood (~font-size-aware for 12pt text)
    #   C=10         → subtract 10 from the local mean to compensate for noise
    #   THRESH_BINARY: pixel > T(p) → 255 (white), else 0 (black)
    binary = cv2.adaptiveThreshold(
        sharpened,
        maxValue   = 255,
        adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType  = cv2.THRESH_BINARY,
        blockSize  = 31,   # must be ODD; increase for larger text/lower DPI
        C          = 10,   # bias: higher C → more pixels become white (lighter image)
    )

    # ── Stage 7: Morphological Closing ────────────────────────────────────────
    # Morphological operations treat the image as a binary set of "on" pixels.
    #
    # CLOSING = DILATION followed by EROSION with the same structuring element.
    #   Dilation: each "on" pixel expands to fill the structuring element shape.
    #   Erosion:  only "on" pixels that fit the full structuring element survive.
    # Net effect of closing: FILLS SMALL GAPS between nearby "on" pixels.
    #
    # Why we need this: adaptive thresholding sometimes breaks thin character
    # strokes (e.g. the crossbar of 't', dots of 'i', serifs) into disconnected
    # fragments. Closing with a 2×2 rectangle reconnects strokes that are ≤1px
    # apart — making characters whole again for the VLM to read.
    #
    # kernel = 2×2 rectangle (MORPH_RECT):
    #   small enough not to merge separate characters,
    #   large enough to close the typical gaps caused by thresholding.
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    # ── Stage 8: Colour Blend (preserve visual context) ───────────────────────
    # The binary image is perfect for text OCR but loses ALL colour information.
    # Problem: VLMs also need to see diagrams, charts, colour-coded annotations.
    # Solution: blend the binarised result with the ORIGINAL colour image.
    #
    # closed_3ch: expand grayscale binary to 3-channel (R=G=B) so shapes match
    # original:   the original colour numpy array (RGB)
    # blend:      0.6 × binary_sharpened + 0.4 × original_colour
    #   → text regions snap to high-contrast black/white
    #   → coloured diagrams retain enough colour for the VLM to understand them
    closed_3ch = cv2.cvtColor(closed, cv2.COLOR_GRAY2RGB)
    blended    = cv2.addWeighted(closed_3ch, 0.6, img_rgb, 0.4, 0)

    # ── Encode result → PNG bytes ─────────────────────────────────────────────
    # We always output PNG (lossless) — JPEG compression would re-introduce
    # artefacts that our pipeline just removed.
    result_pil = Image.fromarray(blended)
    buf        = io.BytesIO()
    result_pil.save(buf, format="PNG", optimize=True)
    processed_bytes = buf.getvalue()

    # Also produce a base64 version for the frontend image preview
    preview_b64 = base64.standard_b64encode(processed_bytes).decode("utf-8")

    return processed_bytes, preview_b64


def _deskew(gray: np.ndarray) -> np.ndarray:
    """
    Detect and correct image skew using the Probabilistic Hough Line Transform.

    Algorithm
    ---------
    1. Canny edge detection → thin edge map
    2. HoughLinesP → find line segments in the edge map
    3. Compute angle of each segment; collect angles in (-15°, +15°) range
    4. Take median angle as the skew estimate
    5. Rotate image by -median_angle to straighten it

    Why the ±15° limit?
        Larger angles are likely intentional layout choices (vertical text,
        rotated diagrams) not skew errors. We only fix small unintentional tilts.

    Parameters
    ----------
    gray : np.ndarray
        Grayscale image (uint8, shape H×W)

    Returns
    -------
    np.ndarray
        Deskewed grayscale image, same dimensions as input.
    """

    # Canny edge detection: two-threshold hysteresis
    #   threshold1=50  → weak edge candidate
    #   threshold2=150 → strong edge (definitely an edge)
    #   apertureSize=3 → Sobel kernel size for gradient computation
    # Result: thin single-pixel edge map
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)

    # Probabilistic Hough Line Transform
    # Unlike standard Hough, HoughLinesP works on a random subset of edge pixels
    # and returns line SEGMENTS (x1,y1,x2,y2) rather than infinite lines (ρ,θ).
    #   rho=1        → distance resolution of the accumulator (1 pixel)
    #   theta=π/180  → angle resolution (1 degree)
    #   threshold=100→ minimum number of votes (intersections) for a line
    #   minLineLength=100 → discard segments shorter than 100px
    #   maxLineGap=10     → merge collinear segments with gap ≤ 10px
    lines = cv2.HoughLinesP(
        edges, rho=1, theta=np.pi / 180,
        threshold=100, minLineLength=100, maxLineGap=10
    )

    if lines is None:
        return gray   # no lines detected — image is already straight (or blank)

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x2 == x1:
            continue   # vertical line — arctan undefined, skip
        # arctan2(Δy, Δx) gives angle in radians; convert to degrees
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
        if -15.0 < angle < 15.0:
            angles.append(angle)   # only collect near-horizontal angles

    if not angles:
        return gray   # no usable angles — return unchanged

    # Median is robust to outlier angles from diagonal diagram lines
    median_angle = float(np.median(angles))

    # Skip rotation if skew is negligible (< 0.5° — within rounding error)
    if abs(median_angle) < 0.5:
        return gray

    # Build 2D rotation matrix around image centre
    h, w = gray.shape
    M = cv2.getRotationMatrix2D(center=(w // 2, h // 2), angle=median_angle, scale=1.0)

    # warpAffine applies the rotation
    #   INTER_CUBIC    → bicubic interpolation (smoother than bilinear for rotation)
    #   BORDER_REPLICATE → fills new border pixels by replicating edge pixels
    #                      (avoids black borders that confuse thresholding)
    rotated = cv2.warpAffine(
        gray, M, (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


# ═════════════════════════════════════════════════════════════════════════════
#  BLOCK 2 — VLM ANALYSIS
#  Purpose: send preprocessed image to Qwen2.5-VL and get structured output.
#  Model  : Qwen/Qwen2.5-VL-7B-Instruct (vision-language transformer, 7B params)
#  API    : OpenAI-compatible chat/completions via HuggingFace router
# ═════════════════════════════════════════════════════════════════════════════

def vlm_analyse(processed_bytes: bytes) -> dict:
    """
    Send preprocessed image bytes to the Vision-Language Model.

    The image is base64-encoded and embedded as a data-URL in the chat message.
    The model returns a JSON string with:
        heading        — short title for the content
        summary        — student-friendly explanation
        extracted_text — raw OCR output
        visual_elements— description of non-text visual content

    Parameters
    ----------
    processed_bytes : bytes
        PNG bytes from the CV preprocessing pipeline.

    Returns
    -------
    dict
        Parsed JSON response from the VLM.
    """

    # Encode image as base64 data-URL
    # Format: "data:<media_type>;base64,<b64_string>"
    # The VLM's tokeniser splits this into image patches internally.
    b64      = base64.standard_b64encode(processed_bytes).decode("utf-8")
    data_url = f"data:image/png;base64,{b64}"

    # Chat Completions API call — multimodal message format:
    #   content is a list of content blocks, mixing image_url and text
    completion = vlm_client.chat.completions.create(
        model      = VLM_MODEL,
        max_tokens = VLM_MAX_TOK,
        messages   = [
            {
                "role": "user",
                "content": [
                    # Block 1: the preprocessed image
                    {
                        "type"     : "image_url",
                        "image_url": {"url": data_url},
                    },
                    # Block 2: the instruction prompt
                    {
                        "type": "text",
                        "text": VLM_PROMPT,
                    },
                ],
            }
        ],
    )

    # Extract the text reply from the first (and only) choice
    raw = completion.choices[0].message.content.strip()

    # Some models wrap JSON in ```json ... ``` — strip those fences if present
    if raw.startswith("```"):
        parts = raw.split("```")
        raw   = parts[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)


# ═════════════════════════════════════════════════════════════════════════════
#  BLOCK 3 — ML / DL POST-PROCESSING
#  Purpose: enrich the VLM output with additional analysis layers.
#
#  3a. TF-IDF Keyword Extraction  (classical ML — scikit-learn)
#      Finds the most discriminative terms in the extracted text.
#
#  3b. Zero-Shot Topic Classification  (DL — BART-MNLI transformer)
#      Labels the content with an academic topic without any training data.
#
#  3c. Readability Score  (rule-based NLP)
#      Flesch Reading Ease — estimates difficulty level of the extracted text.
# ═════════════════════════════════════════════════════════════════════════════

def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    """
    Extract the top N most informative keywords using TF-IDF.

    Approach — single-document TF-IDF trick:
        Normally TF-IDF is computed across a corpus (many documents).
        For a single document we split it into SENTENCES and treat each
        sentence as a "document". Words that appear in one or a few sentences
        (high IDF) AND appear often in those sentences (high TF) rise to the top.

    Parameters
    ----------
    text : str
        The OCR-extracted text from the image.
    top_n : int
        Number of keywords to return (default 10).

    Returns
    -------
    list[str]
        List of top keyword strings, sorted by TF-IDF score descending.
    """

    if not text or len(text.strip()) < 20:
        return []   # not enough text to do meaningful keyword extraction

    # Split text into "documents" (sentences) for TF-IDF
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 5]

    if len(sentences) < 2:
        # Too few sentences — fall back to simple word frequency
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text.lower())
        from collections import Counter
        freq = Counter(words)
        # Remove very common English stop words manually
        stops = {"this","that","with","from","have","been","they","were",
                 "their","will","when","what","also","some","more","into"}
        return [w for w, _ in freq.most_common(top_n + 20)
                if w not in stops][:top_n]

    # TfidfVectorizer parameters:
    #   ngram_range=(1,2)  → include single words AND two-word phrases
    #                         e.g. "neural" and "neural network" both scored
    #   stop_words="english" → remove common words (the, is, at, which, etc.)
    #   max_df=0.85          → ignore terms that appear in >85% of sentences
    #                          (too common to be informative)
    #   min_df=1             → must appear at least once (no cutoff for small docs)
    #   max_features=500     → cap vocabulary size for efficiency
    vectoriser = TfidfVectorizer(
        ngram_range  = (1, 2),
        stop_words   = "english",
        max_df       = 0.85,
        min_df       = 1,
        max_features = 500,
    )

    try:
        # fit_transform:
        #   fit   → build vocabulary, compute IDF scores from sentence corpus
        #   transform → compute TF-IDF matrix (rows=sentences, cols=terms)
        tfidf_matrix = vectoriser.fit_transform(sentences)

        # Sum each column → total importance of each term across all sentences
        # .toarray() converts sparse matrix to dense numpy array
        scores       = tfidf_matrix.sum(axis=0).A1   # shape: (vocab_size,)

        # Map column indices → feature names (terms)
        feature_names = vectoriser.get_feature_names_out()

        # Sort by score descending, take top N
        ranked_idx = scores.argsort()[::-1][:top_n]
        keywords   = [feature_names[i] for i in ranked_idx]
        return keywords

    except ValueError:
        # Vectoriser fails if all tokens are stop-words — return empty
        return []


def classify_topic(text: str) -> tuple[str, float]:
    """
    Use zero-shot NLI classification to assign an academic topic label.

    Model: facebook/bart-large-mnli
    Task:  Given a text and a list of candidate labels, score each label
           via Natural Language Inference (entailment probability).

    For each label L, the model is internally prompted:
        "This text is about {L}"
    and returns the probability that this hypothesis is ENTAILED by the text.

    Parameters
    ----------
    text : str
        The combined heading + summary text to classify.

    Returns
    -------
    topic : str
        The best-matching academic topic label.
    confidence : float
        Confidence score in range [0.0, 1.0].
    """

    if not ZS_AVAILABLE or not text or len(text.strip()) < 20:
        return "General Knowledge", 0.0

    # Truncate to 512 characters — BART has a 1024-token context limit;
    # the labels themselves take tokens, so we leave headroom.
    text_truncated = text[:512]

    try:
        result = zero_shot(
            text_truncated,
            candidate_labels = TOPIC_LABELS,
            # multi_label=False means scores sum to 1 (single best label)
            multi_label      = False,
        )
        # result["labels"][0] is the highest-scoring label
        # result["scores"][0] is its confidence (0-1)
        return result["labels"][0], round(result["scores"][0], 3)
    except Exception as e:
        print(f"Zero-shot classification error: {e}")
        return "General Knowledge", 0.0


def flesch_reading_ease(text: str) -> dict:
    """
    Compute the Flesch Reading Ease score for the extracted text.

    Formula (Flesch, 1948):
        FRE = 206.835 - 1.015 × (words/sentences) - 84.6 × (syllables/words)

    Score interpretation:
        90-100 → Very Easy   (5th grade)
        70-90  → Easy        (6th grade)
        60-70  → Standard    (7th grade)
        50-60  → Fairly Difficult (high school)
        30-50  → Difficult   (college)
        0-30   → Very Confusing (professional/academic)

    Parameters
    ----------
    text : str
        Raw text (OCR output or summary).

    Returns
    -------
    dict with keys: score (float), level (str), avg_sentence_len (float),
                    avg_syllables_per_word (float)
    """

    if not text or len(text.strip()) < 30:
        return {"score": 0.0, "level": "N/A", "avg_sentence_len": 0, "avg_syllables_per_word": 0}

    # Count sentences — split on .!? followed by whitespace or end of string
    sentences = re.split(r'[.!?]+\s*', text.strip())
    sentences = [s for s in sentences if len(s.split()) > 0]
    n_sentences = max(len(sentences), 1)

    # Count words — alphabetic tokens only
    words = re.findall(r'[a-zA-Z]+', text)
    n_words = max(len(words), 1)

    # Count syllables using a simple heuristic:
    #   Count vowel groups (consecutive vowels) per word.
    #   Every word has at least 1 syllable.
    #   Words ending in silent 'e' have that vowel subtracted.
    def count_syllables(word: str) -> int:
        word = word.lower()
        count = len(re.findall(r'[aeiou]+', word))   # vowel groups
        if word.endswith('e') and len(word) > 2:
            count -= 1   # silent final 'e' (e.g. "make", "have")
        return max(count, 1)   # every word has ≥ 1 syllable

    n_syllables = sum(count_syllables(w) for w in words)

    # Flesch Reading Ease formula
    avg_sent_len     = n_words / n_sentences
    avg_syll_per_word = n_syllables / n_words
    score = 206.835 - (1.015 * avg_sent_len) - (84.6 * avg_syll_per_word)
    score = max(0.0, min(100.0, round(score, 1)))

    # Map score to human-readable level
    if score >= 90:   level = "Very Easy"
    elif score >= 70: level = "Easy"
    elif score >= 60: level = "Standard"
    elif score >= 50: level = "Fairly Difficult"
    elif score >= 30: level = "Difficult"
    else:             level = "Very Confusing (Academic)"

    return {
        "score"                 : score,
        "level"                 : level,
        "avg_sentence_len"      : round(avg_sent_len, 1),
        "avg_syllables_per_word": round(avg_syll_per_word, 2),
    }


# ═════════════════════════════════════════════════════════════════════════════
#  BLOCK 3b — TESSERACT BOUNDING BOX EXTRACTION
#  Purpose: get EXACT pixel coordinates for every word on the ORIGINAL image
#           so the frontend can draw perfectly-placed keyword highlights.
#
#  Why Tesseract here instead of the VLM?
#    The VLM (Qwen2.5-VL) returns OCR text as a plain string — no positions.
#    Tesseract's image_to_data() returns a full TSV with one row per word,
#    each with pixel (left, top, width, height) on the original image.
#    We run Tesseract on the ORIGINAL (not preprocessed) image so coordinates
#    map 1:1 to the image the frontend displays to the user.
# ═════════════════════════════════════════════════════════════════════════════

def get_keyword_bounding_boxes(image_bytes: bytes, keywords: list[str]) -> list[dict]:
    """
    Use EasyOCR to detect every word in the image with its exact bounding box,
    then return boxes only for words that match a TF-IDF keyword.

    WHY EasyOCR instead of Tesseract?
    ──────────────────────────────────
    Tesseract requires a separate system binary install (not trivial on Windows).
    EasyOCR is 100% Python — installed via pip, no PATH setup needed.
    It uses a 2-stage deep learning pipeline:
      Stage 1 — CRAFT text detector (CNN): finds text regions as polygons
      Stage 2 — CRNN recogniser: reads characters in each region

    Output format from easyocr.Reader.readtext():
      [ ( [[x1,y1],[x2,y1],[x2,y2],[x1,y2]], "word", confidence ), ... ]
    Each result = (4-corner polygon, recognised text, confidence 0-1).

    We convert the polygon → axis-aligned rect (x, y, w, h) by taking
    min/max of the 4 corners. This is the box drawn on the canvas.

    Matching strategy:
    ──────────────────
    • Case-insensitive: "Python" matches keyword "python"
    • Partial match: detected "algorithms" matches keyword "algorithm"
    • Multi-word keyword "machine learning": if both "machine" and "learning"
      appear on roughly the same Y level (within 20px), their boxes are merged.
    • Confidence filter: skip detections with confidence < 0.3 (noise)

    Parameters
    ----------
    image_bytes : bytes
        Raw ORIGINAL image bytes — NOT preprocessed, so coordinates match
        the image dimensions the frontend will display.
    keywords : list[str]
        TF-IDF keywords from extract_keywords().

    Returns
    -------
    list[dict]
        Each dict: { "keyword": str, "x": int, "y": int, "w": int, "h": int }
        Pixel coordinates in original image space.
    """

    if not OCR_AVAILABLE or not keywords:
        return []

    try:
        # ── Step 1: Decode to numpy array (EasyOCR accepts numpy arrays) ──────
        pil_img  = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np   = np.array(pil_img)   # shape (H, W, 3), dtype uint8

        # ── Step 2: Run EasyOCR text detection + recognition ─────────────────
        # detail=1 → return bounding boxes (detail=0 = text only, no boxes)
        # paragraph=False → return individual words, not merged paragraphs
        # width_ths=0.7 → horizontal merge threshold for adjacent characters
        raw_results = ocr_reader.readtext(
            img_np,
            detail      = 1,
            paragraph   = False,
            width_ths   = 0.7,
        )
        # raw_results: list of (bbox_polygon, text, confidence)
        # bbox_polygon: [[x1,y1],[x2,y1],[x2,y2],[x1,y2]]

        # ── Step 3: Convert polygons → axis-aligned rects ────────────────────
        word_records = []
        for (polygon, text, conf) in raw_results:
            if conf < 0.3 or not text.strip():
                continue   # skip low-confidence detections and blank strings

            # polygon = list of 4 [x, y] points (may be slightly rotated)
            # Axis-aligned rect = bounding box of all 4 corners
            xs = [pt[0] for pt in polygon]
            ys = [pt[1] for pt in polygon]
            x  = int(min(xs))
            y  = int(min(ys))
            w  = int(max(xs) - min(xs))
            h  = int(max(ys) - min(ys))

            word_records.append({
                "text"  : text.strip().lower(),
                "orig"  : text.strip(),
                "x": x, "y": y, "w": w, "h": h,
                "conf"  : conf,
            })

        # ── Step 4: Match keywords → word records ────────────────────────────
        result_boxes = []
        used_indices = set()

        for kw in keywords:
            kw_lower  = kw.lower().strip()
            kw_tokens = kw_lower.split()   # handles multi-word keywords
            n_tokens  = len(kw_tokens)

            if n_tokens == 1:
                # ── Single word keyword ────────────────────────────────────────
                for idx, rec in enumerate(word_records):
                    if idx in used_indices:
                        continue
                    # Match: exact OR detected word starts with keyword
                    # (handles plurals: keyword "algorithm" matches "algorithms")
                    if rec["text"] == kw_lower or rec["text"].startswith(kw_lower):
                        result_boxes.append({
                            "keyword": kw,
                            "x": rec["x"], "y": rec["y"],
                            "w": rec["w"], "h": rec["h"],
                        })
                        used_indices.add(idx)

            else:
                # ── Multi-word keyword: find adjacent words at same Y level ───
                # "machine learning" → find "machine" then "learning" within
                # 20px vertical distance and to the right of "machine"
                for i in range(len(word_records) - n_tokens + 1):
                    if any((i + j) in used_indices for j in range(n_tokens)):
                        continue

                    # Check each token matches in sequence
                    match  = True
                    window = []
                    for j in range(n_tokens):
                        rec = word_records[i + j]
                        tok = kw_tokens[j]
                        if not (rec["text"] == tok or rec["text"].startswith(tok)):
                            match = False
                            break
                        window.append(rec)

                    if not match:
                        continue

                    # Check all matched words are on roughly the same line
                    # (within 20px vertically — handles slight text baseline variation)
                    y_vals = [r["y"] for r in window]
                    if max(y_vals) - min(y_vals) > 20:
                        continue

                    # Merge all word boxes into one spanning bounding box
                    x1 = min(r["x"] for r in window)
                    y1 = min(r["y"] for r in window)
                    x2 = max(r["x"] + r["w"] for r in window)
                    y2 = max(r["y"] + r["h"] for r in window)

                    result_boxes.append({
                        "keyword": kw,
                        "x": x1, "y": y1,
                        "w": x2 - x1, "h": y2 - y1,
                    })
                    for j in range(n_tokens):
                        used_indices.add(i + j)

        return result_boxes

    except Exception as e:
        print(f"EasyOCR bounding box extraction failed: {e}")
        import traceback; traceback.print_exc()
        return []


def run_ml_dl_analysis(vlm_result: dict) -> dict:
    """
    Orchestrate all ML/DL analysis on the VLM's output.

    Parameters
    ----------
    vlm_result : dict
        Parsed JSON from vlm_analyse() containing heading, summary,
        extracted_text, visual_elements.

    Returns
    -------
    dict
        Analysis results to be merged into the final API response.
    """

    extracted_text = vlm_result.get("extracted_text", "")
    summary        = vlm_result.get("summary", "")
    heading        = vlm_result.get("heading", "")

    # Combine heading + summary for topic classification
    # (more context → better classification than text alone)
    classification_input = f"{heading}. {summary}"

    # Run all three analysis layers
    keywords              = extract_keywords(extracted_text, top_n=10)
    topic, confidence     = classify_topic(classification_input)
    readability           = flesch_reading_ease(extracted_text or summary)

    # Word count — useful study metric
    word_count = len(re.findall(r'\b\w+\b', extracted_text))

    return {
        "keywords"   : keywords,
        "topic"      : topic,
        "confidence" : confidence,
        "readability": readability,
        "word_count" : word_count,
    }


# ═════════════════════════════════════════════════════════════════════════════
#  BLOCK 4 — PDF UTILITIES
# ═════════════════════════════════════════════════════════════════════════════

def pdf_page_to_png_bytes(page: fitz.Page) -> bytes:
    """
    Rasterise a single PDF page to PNG bytes using PyMuPDF.

    Parameters
    ----------
    page : fitz.Page
        A PyMuPDF page object from an opened document.

    Returns
    -------
    bytes
        PNG-encoded pixel data of the rendered page.

    Notes
    -----
    fitz.Matrix(2, 2) = 2× zoom = 144 DPI (PDF native is 72 DPI).
    Higher DPI → more pixels → better OCR at the cost of larger payload.
    144 DPI is the sweet spot for text clarity vs. API size limits.
    """
    mat = fitz.Matrix(2, 2)        # 2× zoom in both X and Y
    pix = page.get_pixmap(matrix=mat)
    return pix.tobytes("png")


# ═════════════════════════════════════════════════════════════════════════════
#  BLOCK 5 — FLASK API ROUTES
# ═════════════════════════════════════════════════════════════════════════════

@app.route("/api/analyse/image", methods=["POST"])
def analyse_image():
    """
    POST /api/analyse/image

    Accepts a multipart/form-data upload with field name 'file'.
    Runs the full pipeline:
        CV preprocessing → VLM analysis → ML/DL analysis

    Returns JSON:
    {
        "success"           : true,
        "type"              : "image",
        "result": {
            "heading"       : str,
            "summary"       : str,
            "extracted_text": str,
            "visual_elements": str,
            "keywords"      : [str, ...],
            "topic"         : str,
            "confidence"    : float,
            "readability"   : { score, level, ... },
            "word_count"    : int,
            "preprocessed_image_b64": str   ← base64 PNG for frontend preview
        }
    }
    """
    # ── Validate request ──────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"error": "No file field in request. Use multipart/form-data with field 'file'."}), 400

    f = request.files["file"]

    if f.content_type not in ALLOWED_IMAGE_TYPES:
        return jsonify({"error": f"Unsupported MIME type '{f.content_type}'. Allowed: {ALLOWED_IMAGE_TYPES}"}), 400

    image_bytes = f.read()

    try:
        # ── Stage 0: Save ORIGINAL image as base64 ────────────────────────────
        # We keep the RAW original and send it to the frontend.
        # The frontend draws keyword highlights ON this original image via
        # HTML Canvas API — yellow highlight boxes over the unprocessed photo.
        # Readability is from OCR text (not the image), so preprocessing has
        # no effect on the readability score — it measures TEXT complexity.
        original_b64  = base64.standard_b64encode(image_bytes).decode("utf-8")
        original_mime = f.content_type   # "image/jpeg" or "image/png" etc.

        # ── Stage 1: CV preprocessing ─────────────────────────────────────────
        # 7-stage pipeline cleans the image for the VLM to read accurately.
        # Returns: processed PNG bytes (sent to VLM) + base64 (side-by-side UI)
        processed_bytes, preprocessed_b64 = preprocess_image(image_bytes)

        # ── Stage 2: VLM analysis ─────────────────────────────────────────────
        # Send the PREPROCESSED image to Qwen2.5-VL for best OCR accuracy.
        vlm_result = vlm_analyse(processed_bytes)

        # ── Stage 3: ML/DL enrichment ─────────────────────────────────────────
        # TF-IDF keywords + BART-MNLI topic + Flesch readability on OCR TEXT.
        ml_result = run_ml_dl_analysis(vlm_result)

        # ── Stage 4: Tesseract bounding boxes ────────────────────────────────
        # Run Tesseract on the ORIGINAL image to get exact pixel coordinates
        # for each keyword. The frontend uses these to draw precise yellow
        # highlight boxes — one box per word occurrence, perfectly sized.
        # Note: we pass image_bytes (original) NOT processed_bytes, so the
        # coordinates match the original image dimensions shown to the user.
        word_boxes = get_keyword_bounding_boxes(image_bytes, ml_result.get("keywords", []))

        # ── Merge and return all results ──────────────────────────────────────
        final_result = {
            **vlm_result,                           # heading, summary, OCR, visual_elements
            **ml_result,                            # keywords, topic, confidence, readability
            "word_boxes"             : word_boxes,          # exact Tesseract bounding boxes
            "preprocessed_image_b64" : preprocessed_b64,   # CV-cleaned (side-by-side preview)
            "original_image_b64"     : original_b64,        # raw original → Canvas highlight
            "original_image_mime"    : original_mime,       # needed for correct data-URL
        }

        return jsonify({"success": True, "type": "image", "result": final_result})

    except json.JSONDecodeError as e:
        return jsonify({"error": f"VLM returned malformed JSON: {e}"}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/analyse/pdf", methods=["POST"])
def analyse_pdf():
    """
    POST /api/analyse/pdf

    Accepts a PDF upload. Rasterises each page, runs the full pipeline
    on each page, then generates an overall document summary.
    Limited to 10 pages to stay within API rate limits.

    Returns JSON:
    {
        "success"       : true,
        "type"          : "pdf",
        "total_pages"   : int,
        "analysed_pages": int,
        "overall"       : { heading, summary, keywords, topic, ... },
        "pages"         : [ { page, heading, summary, ..., preprocessed_image_b64 }, ... ]
    }
    """
    if "file" not in request.files:
        return jsonify({"error": "No file field in request."}), 400

    f = request.files["file"]
    if f.content_type != ALLOWED_PDF_TYPE:
        return jsonify({"error": f"Expected application/pdf, got '{f.content_type}'."}), 400

    pdf_bytes = f.read()

    try:
        # Open PDF from in-memory bytes (no temp file needed)
        doc         = fitz.open(stream=pdf_bytes, filetype="pdf")
        total_pages = len(doc)
        max_pages   = min(total_pages, 10)   # cap at 10 to avoid API throttling
        pages_results = []

        for page_num in range(max_pages):
            page = doc[page_num]

            # ── Per-page pipeline ─────────────────────────────────────────
            # 1. Rasterise PDF page to raw PNG at 144 DPI (2x zoom)
            raw_png = pdf_page_to_png_bytes(page)

            # 2. Save raw PNG as base64 → frontend uses this for Canvas highlight
            page_original_b64 = base64.standard_b64encode(raw_png).decode("utf-8")

            # 3. CV preprocessing (clean up for VLM OCR accuracy)
            processed_bytes, preprocessed_b64 = preprocess_image(raw_png)

            # 4. VLM analysis on preprocessed image
            vlm_result = vlm_analyse(processed_bytes)

            # 5. ML/DL: TF-IDF keywords, zero-shot topic, readability (from OCR text)
            ml_result = run_ml_dl_analysis(vlm_result)

            # Tesseract bounding boxes on the raw page PNG
            page_word_boxes = get_keyword_bounding_boxes(raw_png, ml_result.get("keywords", []))

            pages_results.append({
                "page"                   : page_num + 1,    # 1-indexed for humans
                **vlm_result,
                **ml_result,
                "word_boxes"             : page_word_boxes,    # exact keyword positions
                "preprocessed_image_b64" : preprocessed_b64,  # CV-cleaned (preview)
                "original_image_b64"     : page_original_b64,  # raw → Canvas highlight
                "original_image_mime"    : "image/png",         # PDF pages always PNG
            })

        # ── Overall document summary ──────────────────────────────────────────
        # Concatenate extracted text from all pages, truncate to ~6000 chars
        # to stay within the VLM's context window.
        combined_text = "\n\n".join(
            p.get("extracted_text", "") for p in pages_results
        )[:6000]

        overall_prompt = (
            f"This is the full text of a multi-page academic document:\n\n{combined_text}\n\n"
            "Write a document-level heading (max 10 words) and a comprehensive summary "
            "(200-300 words) that a student can use as study notes.\n"
            "Respond ONLY in this exact JSON with no markdown:\n"
            '{"heading":"...","summary":"..."}'
        )

        overall_completion = vlm_client.chat.completions.create(
            model      = VLM_MODEL,
            max_tokens = 512,
            messages   = [{"role": "user", "content": overall_prompt}],
        )
        raw_overall = overall_completion.choices[0].message.content.strip()
        if raw_overall.startswith("```"):
            parts       = raw_overall.split("```")
            raw_overall = parts[1]
            if raw_overall.startswith("json"):
                raw_overall = raw_overall[4:]
            raw_overall = raw_overall.strip()
        overall_vlm = json.loads(raw_overall)

        # Run ML/DL analysis on the overall document text too
        overall_ml = run_ml_dl_analysis({
            "extracted_text": combined_text,
            "summary"       : overall_vlm.get("summary", ""),
            "heading"       : overall_vlm.get("heading", ""),
        })

        return jsonify({
            "success"       : True,
            "type"          : "pdf",
            "total_pages"   : total_pages,
            "analysed_pages": max_pages,
            "overall"       : {**overall_vlm, **overall_ml},
            "pages"         : pages_results,
        })

    except json.JSONDecodeError as e:
        return jsonify({"error": f"VLM returned malformed JSON: {e}"}), 500
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/health", methods=["GET"])
def health():
    """GET /api/health — liveness check, returns pipeline component status."""
    return jsonify({
        "status"    : "ok",
        "vlm_model" : VLM_MODEL,
        "cv_pipeline": [
            "upscale (LANCZOS)", "deskew (HoughLinesP)", "denoise (Gaussian 3×3)",
            "CLAHE (clip=2, tile=8×8)", "sharpen (unsharp mask σ=3)",
            "binarise (adaptive threshold block=31)", "morph close (2×2 rect)",
            "colour blend (60/40)",
        ],
        "ml_features": [
            "TF-IDF keyword extraction (scikit-learn)",
            f"Zero-shot topic classification (BART-MNLI, available={ZS_AVAILABLE})",
            "Flesch Reading Ease score",
            "Word count",
        ],
    })


# ═════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 65)
    print("  SnapWise Backend — Interview-Grade Screenshot Analyser")
    print("=" * 65)
    print(f"  VLM Model    : {VLM_MODEL}")
    print(f"  HF Token     : {'SET ✓' if HF_TOKEN != 'hf_PASTE_YOUR_TOKEN_HERE' else 'NOT SET ✗  →  edit line 13'}")
    print(f"  Zero-shot DL : {'AVAILABLE ✓' if ZS_AVAILABLE else 'UNAVAILABLE (install transformers)'}")
    print(f"  EasyOCR      : {'AVAILABLE ✓' if OCR_AVAILABLE else 'UNAVAILABLE (run pip install easyocr)'}")
    print(f"  CV Pipeline  : OpenCV {cv2.__version__}")
    print("  Routes       : POST /api/analyse/image")
    print("                 POST /api/analyse/pdf")
    print("                 GET  /api/health")
    print("=" * 65)
    app.run(debug=True, port=5000)
