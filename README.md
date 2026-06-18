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
- `qwen25_vl_7b` - Qwen2.5-VL 7B via OpenRouter-compatible chat completions;
- `qwen25_vl_32b` - Qwen2.5-VL 32B via OpenRouter-compatible chat completions;
- `qwen25_vl_72b` - Qwen2.5-VL 72B via OpenRouter-compatible chat completions;
- `gemma3_vision` - Gemma 3 Vision via OpenRouter-compatible chat completions;
- `internvl` - InternVL via OpenRouter-compatible chat completions;
- `minicpm_v` - MiniCPM-V via OpenRouter-compatible chat completions;
- `florence2` - local Microsoft Florence-2 through Hugging Face Transformers.

For Russian handwritten text, Yandex and other cloud OCR systems are good baselines to benchmark first. EasyOCR and PaddleOCR are useful local baselines with Cyrillic support, but should be measured on your own handwriting samples. TrOCR is included for experimentation, but the default public handwritten model is not Russian-specific; it will likely need a Cyrillic or project-specific fine-tuned model.

The VLM providers are not classic OCR engines. They send the image with this prompt by default:

```text
Transcribe this Russian handwritten text exactly. Preserve line breaks. Return only the transcribed text.
```

OpenRouter model identifiers can change by provider availability, so the defaults in `.env.example` are meant as starting points. Override them in `.env` when a provider exposes a different model id.

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
  --providers yandex,tesseract,easyocr,paddleocr,trocr,qwen25_vl_7b,gemma3_vision,minicpm_v,florence2
```

The script writes a CSV report with recognized text, runtime, CER, and WER for each provider.

Recommended first test set:

- 30-50 Russian handwritten samples from different writers;
- the same pages photographed under different lighting conditions;
- ground truth typed manually in `.txt` files;
- separate groups for clean scans, phone photos, tilted pages, and low contrast.

Use CER to track character-level improvements and WER to track word-level readability.

Suggested research table:

| Provider | Type | CER | WER | Speed | Cost |
| --- | --- | --- | --- | --- | --- |
| `tesseract` | OCR |  |  |  |  |
| `easyocr` | OCR |  |  |  |  |
| `yandex` | OCR API |  |  |  |  |
| `trocr` | Transformer OCR |  |  |  |  |
| `qwen25_vl_7b` | VLM |  |  |  |  |
| `qwen25_vl_32b` | VLM |  |  |  |  |
| `qwen25_vl_72b` | VLM |  |  |  |  |
| `gemma3_vision` | VLM |  |  |  |  |
| `internvl` | VLM |  |  |  |  |
| `minicpm_v` | VLM |  |  |  |  |
| `florence2` | VLM / document understanding |  |  |  |  |

For the educational task, evaluate two layers separately:

- OCR/VLM to text: CER, WER, speed, and cost;
- OCR/VLM to error detection: whether the final grammar analysis finds the real student mistakes.

This second layer matters because an OCR system with worse WER may still preserve the mistakes that are important for feedback.

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
