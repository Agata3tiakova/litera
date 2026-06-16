# Litera

A Flask web application for recognizing handwritten text from an image and analyzing writing quality with AI: spelling, punctuation, style, vocabulary, and short improvement recommendations.

## Features

- image upload with handwritten text;
- image preprocessing before OCR;
- text recognition via pluggable OCR providers;
- editable recognized text before analysis;
- grammar and style analysis via YandexGPT;
- result storage in SQLite;
- analysis page with a downloadable text report.

## Tech Stack

- Python / Flask
- Flask-SQLAlchemy
- SQLite
- OpenCV
- Yandex Vision API
- YandexGPT
- Bootstrap

## OCR Providers

The app uses `OCR_PROVIDER=yandex` by default. OCR engines are implemented as interchangeable providers:

- `yandex` - current Yandex Vision integration;
- `tesseract` - local Tesseract via `pytesseract`;
- `easyocr` - local EasyOCR with Russian and English languages;
- `paddleocr` - local PaddleOCR with Cyrillic model settings;
- `trocr` - Hugging Face TrOCR model interface;
- `google_vision` - Google Cloud Vision OCR via REST API;
- `azure_vision` - Azure Vision Read OCR via REST API.

For Russian handwritten text, Yandex and other cloud OCR systems are good baselines to benchmark first. EasyOCR and PaddleOCR are useful local baselines with Cyrillic support, but should be measured on your own handwriting samples. TrOCR is included for experimentation, but the default public handwritten model is not Russian-specific; it will likely need a Cyrillic or project-specific fine-tuned model.

## Local Setup

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create `.env` from `.env.example` and fill in the required API credentials.
4. Run the application:

```bash
python run.py
```

The application will be available at `http://127.0.0.1:5000`.

## OCR Benchmark

Optional OCR dependencies are separated from the main app because they are large:

```bash
pip install -r requirements-ocr.txt
```

Tesseract also requires the native Tesseract binary and Russian language data installed on the system.

Prepare a dataset with matching image and text files:

```text
data/ocr/images/sample_001.jpg
data/ocr/ground_truth/sample_001.txt
```

Run a benchmark:

```bash
python scripts/ocr_benchmark.py ^
  --images data/ocr/images ^
  --ground-truth data/ocr/ground_truth ^
  --providers yandex,tesseract,easyocr,paddleocr,trocr
```

The script writes a CSV report with recognized text, runtime, CER, and WER for each provider.

Recommended first test set:

- 30-50 Russian handwritten samples from different writers;
- the same pages photographed under different lighting conditions;
- ground truth typed manually in `.txt` files;
- separate groups for clean scans, phone photos, tilted pages, and low contrast.

Use CER to track character-level improvements and WER to track word-level readability.

## Project Structure

- `app/__init__.py` - Flask application factory;
- `app/config.py` - application configuration;
- `app/routes.py` - web pages and API routes;
- `app/models.py` - database models;
- `app/services/ocr/` - OCR provider implementations;
- `app/services/` - image preprocessing, text analysis, and Yandex IAM integration;
- `app/templates/` - HTML templates;
- `app/static/` - CSS and JavaScript assets.

## Notes

The `.env` file, `sa-key.json`, local database, uploaded files, IDE settings, and virtual environment are excluded from git via `.gitignore`.
