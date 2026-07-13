# Litera

Litera is a school project: a Flask web application that recognizes a student's handwritten Russian text from an image and then analyzes writing quality with AI: spelling, punctuation, style, vocabulary, and short improvement recommendations.

Russian handwritten text recognition turned out to be the hardest part of the project, especially for school notebook photos. Because of that, the project includes a separate OCR/VLM benchmark and research workflow comparing classic OCR systems, OCR APIs, Transformer OCR models, and vision-language models.

Research report: [OCR Research: Russian Handwritten School Text](docs/ocr_research.md).

## Features

- image upload with handwritten text;
- image preprocessing before OCR;
- text recognition via pluggable OCR providers;
- editable recognized text before analysis;
- grammar and style analysis via YandexGPT;
- result storage in SQLite;
- analysis page with a downloadable text report.

## Tech Stack

| Area | Technologies |
| --- | --- |
| Web application | Python, Flask, Flask-SQLAlchemy |
| Database | SQLite |
| Image processing | OpenCV |
| OCR provider layer | Yandex Vision API, Tesseract, EasyOCR |
| Transformer OCR / VLM research | Hugging Face Transformers, TrOCR, Cyrillic TrOCR, Florence-2, OpenRouter-compatible VLMs |
| Text analysis | YandexGPT |
| Frontend | Bootstrap |

`Yandex Vision API` and `YandexGPT` are part of the application runtime: Yandex Vision can be used as the default OCR provider, and YandexGPT is used for grammar/style analysis after OCR. Other OCR and VLM systems are connected through the benchmark/provider layer and are listed below.

## OCR/VLM Providers

The app uses `OCR_PROVIDER=yandex` by default. OCR engines are implemented as interchangeable providers:

| Provider | Type | Notes |
| --- | --- | --- |
| `yandex` | OCR API | Yandex Vision integration and current default provider. |
| `tesseract` | Local OCR | Local Tesseract via `pytesseract`. |
| `easyocr` | Local OCR | Local EasyOCR with Russian and English language support. |
| `trocr` | Transformer OCR | Generic Hugging Face TrOCR interface. |
| `cyrillic_trocr` | Transformer OCR | `cyrillic-trocr/trocr-handwritten-cyrillic`, fine-tuned for Cyrillic handwriting. |
| `qwen25_vl_7b` | VLM | Qwen2.5-VL 7B via OpenRouter-compatible chat completions. |
| `qwen25_vl_32b` | VLM | Qwen2.5-VL 32B via OpenRouter-compatible chat completions. |
| `qwen25_vl_72b` | VLM | Qwen2.5-VL 72B via OpenRouter-compatible chat completions. |
| `gemma3_vision` | VLM | Gemma 3 Vision via OpenRouter-compatible chat completions. |
| `internvl` | VLM | InternVL via OpenRouter-compatible chat completions. |
| `minicpm_v` | VLM | MiniCPM-V via OpenRouter-compatible chat completions. |
| `florence2` | Document/VLM baseline | Local Microsoft Florence-2 through Hugging Face Transformers. |

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

The full benchmark summary, metrics, OCR vs VLM comparison, and conclusions are documented in [docs/ocr_research.md](docs/ocr_research.md).

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
Use `preserve_student_errors_strict` when the experiment is focused on preserving student spelling mistakes rather than maximizing clean-text readability.

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

To measure whether OCR preserves real student mistakes instead of silently correcting them, reuse existing benchmark CSV files:

```bash
python scripts/ocr_student_mistake_preservation.py ^
  --benchmarks tmp/ocr_test_014_final_working_9of9/working_models_dual_ref.csv,tmp/ocr_test_018_yandex_9sample/yandex_dual_ref.csv,tmp/ocr_test_017_cyrillic_trocr/cyrillic_trocr_dual_ref.csv ^
  --student-text data/ocr/student_text ^
  --correct-text data/ocr/correct_text ^
  --output tmp/ocr_student_mistake_preservation.csv ^
  --report tmp/ocr_student_mistake_preservation.md
```

This analysis does not run OCR again. It compares `student_text` with `correct_text`, finds real student-vs-correct differences, and checks whether each OCR output preserved the student variant, corrected it, or lost the word.

To identify places where the best OCR provider may have silently corrected a student mistake, build a preservation risk report:

```bash
python scripts/ocr_preservation_risk_report.py ^
  --preservation tmp/ocr_student_mistake_preservation.csv ^
  --base-provider qwen25_vl_72b ^
  --support-providers yandex,cyrillic_trocr,gemma3_vision ^
  --output tmp/ocr_preservation_risk.csv ^
  --report tmp/ocr_preservation_risk.md
```

This is a preservation-aware ensemble signal. It does not automatically replace the base OCR text. It flags places where the base provider output looks corrected or lost, while another provider preserved the student spelling.

## Research Summary

Full metrics, OCR vs VLM comparison, student-mistake preservation analysis, strict prompt results, and conclusions are documented in [docs/ocr_research.md](docs/ocr_research.md).

Current high-level findings:

- VLMs produce the best CER/WER on Russian handwritten school text because they use visual and language context together.
- Classic OCR systems remain useful baselines, but they lose many handwritten words on notebook photos.
- `qwen25_vl_72b` is the strongest transcription baseline by CER/WER.
- `gemma3_vision` is the second-best VLM baseline.
- `yandex` is the strongest OCR/API baseline and is faster than VLMs, but less accurate on difficult handwriting.
- Raw OCR accuracy is not enough for this educational task because VLMs can silently correct student mistakes.
- The `preserve_student_errors_strict` prompt and preservation-risk reports were added to improve recognition while protecting real student mistakes before LLM analysis.

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
