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
- `trocr` - Hugging Face TrOCR model interface;
- `cyrillic_trocr` - Hugging Face TrOCR model fine-tuned for Cyrillic handwriting;
- `qwen25_vl_7b` - Qwen2.5-VL 7B via OpenRouter-compatible chat completions;
- `qwen25_vl_32b` - Qwen2.5-VL 32B via OpenRouter-compatible chat completions;
- `qwen25_vl_72b` - Qwen2.5-VL 72B via OpenRouter-compatible chat completions;
- `gemma3_vision` - Gemma 3 Vision via OpenRouter-compatible chat completions;
- `internvl` - InternVL via OpenRouter-compatible chat completions;
- `minicpm_v` - MiniCPM-V via OpenRouter-compatible chat completions;
- `florence2` - local Microsoft Florence-2 through Hugging Face Transformers.

For Russian handwritten text, Yandex and other cloud OCR systems are good baselines to benchmark first. EasyOCR is a useful local baseline with Cyrillic support, but should be measured on your own handwriting samples. The default `trocr` model is not Russian-specific; `cyrillic_trocr` uses `cyrillic-trocr/trocr-handwritten-cyrillic`, a Cyrillic handwriting checkpoint for Russian, Ukrainian, and Church Slavonic. The Cyrillic TrOCR provider segments pages into line crops by default because TrOCR checkpoints are trained for line-level recognition, not full-page transcription.

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
data/ocr/student_text/sample_001.txt
data/ocr/correct_text/sample_001.txt
```

Use `student_text` as the primary OCR reference: it should preserve what the student actually wrote, including spelling and punctuation mistakes. Use `correct_text` only as an optional clean reference.

Run a benchmark:

```bash
python scripts/ocr_benchmark.py ^
  --images data/ocr/images ^
  --ground-truth data/ocr/student_text ^
  --clean-ground-truth data/ocr/correct_text ^
  --providers yandex,tesseract,easyocr,trocr,cyrillic_trocr,qwen25_vl_7b,gemma3_vision,minicpm_v,florence2
```

The script writes a CSV report with recognized text, runtime, CER, and WER for each provider. Main CER/WER metrics compare OCR output with `student_text`. `clean_*` metrics compare the same OCR output with `correct_text` when provided.

For `cyrillic_trocr`, use a CUDA-enabled PyTorch build when possible. The provider defaults to `CYRILLIC_TROCR_SEGMENT_LINES=true`, recognizes line crops separately, and joins them with line breaks:

```bash
python scripts/ocr_benchmark.py ^
  --images data/ocr/images ^
  --ground-truth data/ocr/student_text ^
  --clean-ground-truth data/ocr/correct_text ^
  --providers cyrillic_trocr ^
  --keep-original ^
  --output tmp/ocr_cyrillic_trocr_results.csv ^
  --analysis-output tmp/ocr_cyrillic_trocr_analysis.md
```

To compare preprocessing strategies, pass image variants explicitly:

```bash
python scripts/ocr_benchmark.py ^
  --images data/ocr/images ^
  --ground-truth data/ocr/student_text ^
  --clean-ground-truth data/ocr/correct_text ^
  --providers yandex,qwen25_vl_72b,gemma3_vision,tesseract ^
  --keep-original ^
  --variants original,grayscale_soft,contrast,denoise,deskew,line_removed,otsu_binary ^
  --output tmp/ocr_benchmark_results.csv ^
  --analysis-output tmp/ocr_benchmark_analysis.md
```

Available variants:

- `original` - unchanged source image;
- `grayscale_soft` - grayscale without hard thresholding;
- `contrast` - local contrast enhancement with CLAHE;
- `denoise` - light color denoising;
- `deskew` - automatic rotation correction for small skew angles;
- `line_removed` - experimental notebook line removal;
- `otsu_binary` - hard black-and-white thresholding, useful as a baseline.

The optional Markdown analysis report summarizes full-text OCR errors: word substitutions, missing words, extra words, punctuation differences, line-break differences, and hyphenated line breaks.

For VLM providers, you can compare transcription prompts:

```bash
python scripts/ocr_benchmark.py ^
  --images data/ocr/images ^
  --ground-truth data/ocr/student_text ^
  --providers qwen25_vl_72b,gemma3_vision ^
  --keep-original ^
  --variants original ^
  --prompt-variants default,exact_no_correction,school_notebook,literal_uncertain ^
  --output tmp/ocr_prompt_benchmark.csv ^
  --analysis-output tmp/ocr_prompt_benchmark.md
```

Prompt variants are applied only to VLM providers. Classic OCR providers use their normal settings.
OpenRouter requests are retried on temporary rate limits and server errors. Tune `OPENROUTER_MAX_RETRIES` and `OPENROUTER_RETRY_DELAY` in `.env` if prompt/model sweeps hit provider limits.

To measure model stability, repeat each run:

```bash
python scripts/ocr_benchmark.py ^
  --images data/ocr/images ^
  --ground-truth data/ocr/student_text ^
  --providers qwen25_vl_72b ^
  --keep-original ^
  --repeats 3 ^
  --output tmp/ocr_repeated_benchmark.csv ^
  --analysis-output tmp/ocr_repeated_benchmark.md
```

Repeated reports include `repeat_index` in CSV and aggregate min/max/std values for normalized CER in Markdown.

To build a weighted consensus report from an existing benchmark CSV:

```bash
python scripts/ocr_ensemble.py ^
  --input tmp/ocr_benchmark_results.csv ^
  --output-dir tmp/ocr_ensemble ^
  --ground-truth data/ocr/student_text ^
  --clean-ground-truth data/ocr/correct_text
```

The ensemble script uses the best provider as the base text, weights other providers by observed normalized CER, keeps text where stronger models agree, and writes `word_confidence.csv` to flag words and pages that need manual or follow-up OCR review.

To turn word confidence into second-pass OCR tasks:

```bash
python scripts/ocr_followup_tasks.py ^
  --confidence tmp/ocr_ensemble/word_confidence.csv ^
  --output tmp/ocr_followup_tasks.md ^
  --rerun-list tmp/ocr_rerun_images.txt
```

The follow-up report groups low-confidence words by image and recommends whether to rerun the full page or review only listed words.

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
| `cyrillic_trocr` | Transformer OCR |  |  |  |  |
| `qwen25_vl_7b` | VLM |  |  |  |  |
| `qwen25_vl_32b` | VLM |  |  |  |  |
| `qwen25_vl_72b` | VLM |  |  |  |  |
| `gemma3_vision` | VLM |  |  |  |  |
| `internvl` | VLM |  |  |  |  |
| `minicpm_v` | VLM |  |  |  |  |
| `florence2` | VLM / document understanding |  |  |  |  |

Current 9-sample Russian handwriting benchmark, using student text as the primary reference:

| Provider | Type | Avg normalized CER | Avg normalized WER | Avg time |
| --- | --- | ---: | ---: | ---: |
| `qwen25_vl_72b` | VLM | 0.133 | 0.277 | 6.129s |
| `gemma3_vision` | VLM | 0.271 | 0.484 | 4.006s |
| `cyrillic_trocr` | Transformer OCR | 0.686 | 0.958 | 296.509s |
| `easyocr` | OCR | 0.883 | 1.133 | 17.974s |
| `tesseract` | OCR | 0.962 | 1.455 | 3.013s |
| `trocr` | Transformer OCR | 0.978 | 1.000 | 5.780s |

On this sample, `cyrillic_trocr` improves over the generic TrOCR baseline, but it often produces Church Slavonic-like tokens and remains much weaker than the VLM providers. Its result should be treated as a line-level OCR baseline, not as a leading candidate for the current full-page school handwriting task.

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
