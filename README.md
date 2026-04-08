# SnapWise 📸

AI-powered screenshot & PDF analyser for students — built with React + Flask + HuggingFace Inference API.

## Features

- 🖼️ **Screenshot Analysis** — Upload any screenshot; the vision model reads all text (OCR), describes visuals, generates a heading and study-ready summary
- 📄 **PDF Scanning** — Upload a PDF; each page is rendered and analysed; an overall document summary is generated
- 🤖 **HuggingFace AI** — `Qwen/Qwen2.5-VL-7B-Instruct` via HuggingFace Inference API (Hyperbolic provider) powers all OCR, image captioning, and summarisation
- 🎨 **Beautiful UI** — Dark editorial design with animated orbs and smooth transitions

---

## Project Structure

```
snapwise/
├── backend/
│   ├── app.py           ← Flask API
│   └── requirements.txt
└── frontend/
    ├── index.html
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        └── App.css
```

---

## Setup & Run

### 1. Backend (Flask)

```bash
cd backend

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your HuggingFace token (get one at https://huggingface.co/settings/tokens)
export HF_TOKEN=hf_...   # Windows: set HF_TOKEN=hf_...

# Run the Flask server
python app.py
# → Running on http://localhost:5000
```

### 2. Frontend (React + Vite)

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
# → Running on http://localhost:3000
```

Open **http://localhost:3000** in your browser.

---

## Getting Your HuggingFace Token

1. Go to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
2. Click **New token**
3. Give it a name, select **Read** permissions (or **Inference** for serverless)
4. Copy the token starting with `hf_...`
5. Set it as the `HF_TOKEN` environment variable

---

## Model Used

| Model | Provider | Task |
|-------|----------|------|
| `Qwen/Qwen2.5-VL-7B-Instruct` | Hyperbolic (via HF) | Vision + OCR + Summarisation |

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check |
| POST | `/api/analyse/image` | Analyse a screenshot (multipart `file`) |
| POST | `/api/analyse/pdf` | Scan a PDF (multipart `file`) |

### Image response
```json
{
  "success": true,
  "type": "image",
  "result": {
    "heading": "Machine Learning Gradient Descent Explained",
    "summary": "This screenshot covers...",
    "extracted_text": "All OCR text here...",
    "visual_elements": "A graph showing loss curve..."
  }
}
```

### PDF response
```json
{
  "success": true,
  "type": "pdf",
  "total_pages": 12,
  "analysed_pages": 10,
  "overall": { "heading": "...", "summary": "..." },
  "pages": [
    { "page": 1, "heading": "...", "summary": "...", "extracted_text": "..." }
  ]
}
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Your HuggingFace access token (required) |

---

## Tech Stack

- **Frontend**: React 18, Vite
- **Backend**: Flask, Flask-CORS
- **AI**: Qwen2.5-VL-7B-Instruct via HuggingFace Inference API
- **PDF**: PyMuPDF (fitz)
- **Image**: Pillow

---

## Notes

- PDFs are limited to 10 pages per analysis
- Supported image types: JPEG, PNG, GIF, WebP
- Backend runs on port 5000, frontend on port 3000
