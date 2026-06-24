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
| `yandex` | OCR API | 0.417 | 0.609 | 2.674s | Paid API |
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
| `yandex` | OCR API | 0.417 | 0.609 | 2.674s |
| `cyrillic_trocr` | Transformer OCR | 0.686 | 0.958 | 296.509s |
| `easyocr` | OCR | 0.883 | 1.133 | 17.974s |
| `tesseract` | OCR | 0.962 | 1.455 | 3.013s |
| `trocr` | Transformer OCR | 0.978 | 1.000 | 5.780s |

On this sample, `qwen25_vl_72b` is the strongest model, followed by `gemma3_vision`. `yandex` is the strongest classic OCR/API baseline in the 9-sample run and is faster than the VLM providers, but its error rate is still clearly higher on difficult Russian handwriting. `cyrillic_trocr` improves over the generic TrOCR baseline, but it often produces Church Slavonic-like tokens and remains much weaker than the VLM providers. Its result should be treated as a line-level OCR baseline, not as a leading candidate for the current full-page school handwriting task.

Research progress so far:

- switched the primary OCR metric from corrected text to manually checked `student_text`, preserving student spelling mistakes;
- kept `correct_text` as a secondary clean-reference metric only;
- tested preprocessing variants and found that they do not reliably improve strong OCR/VLM providers on the current samples;
- added repeated runs and confidence/ensemble reports to identify pages that need targeted reruns;
- added `cyrillic_trocr` with line segmentation as a Cyrillic Transformer OCR baseline;
- added the paid `yandex` 9-sample run to compare cloud OCR against local OCR and VLM providers on the same dataset;
- added a student-mistake preservation metric to separate raw OCR accuracy from whether the system keeps the student's actual spelling mistakes.

Student mistake preservation on the current 9-sample set:

| Provider | Checked mistakes | Preserved | Corrected | Lost | Preservation rate | Correction rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yandex` | 47 | 12 | 10 | 25 | 0.255 | 0.213 |
| `cyrillic_trocr` | 47 | 9 | 0 | 38 | 0.191 | 0.000 |
| `qwen25_vl_72b` | 47 | 3 | 28 | 16 | 0.064 | 0.596 |
| `gemma3_vision` | 47 | 2 | 21 | 24 | 0.043 | 0.447 |
| `tesseract` | 47 | 1 | 0 | 46 | 0.021 | 0.000 |
| `easyocr` | 47 | 0 | 0 | 47 | 0.000 | 0.000 |
| `trocr` | 47 | 0 | 0 | 47 | 0.000 | 0.000 |

This shows a separate research tradeoff: `qwen25_vl_72b` is best at transcription accuracy, but it often normalizes student mistakes to the clean text. For educational error detection, the next OCR prompt/ensemble step should explicitly optimize mistake preservation, not only CER/WER.

Strict VLM prompt results on the same 9-sample set:

| Provider | Prompt | Avg Norm CER | Avg Norm WER | Preserved mistakes | Corrected mistakes | Preservation rate | Correction rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen25_vl_72b` | default | 0.133 | 0.277 | 3 | 28 | 0.064 | 0.596 |
| `qwen25_vl_72b` | `preserve_student_errors_strict` | 0.109 | 0.290 | 8 | 20 | 0.170 | 0.426 |
| `gemma3_vision` | default | 0.271 | 0.484 | 2 | 21 | 0.043 | 0.447 |
| `gemma3_vision` | `preserve_student_errors_strict` | 0.207 | 0.391 | 8 | 14 | 0.170 | 0.298 |

The strict prompt improved mistake preservation for both VLM providers and also improved normalized CER. It should be treated as the preferred VLM prompt for the educational OCR task, with the caveat that Qwen's normalized WER rose slightly and both models became slower.

Preservation-aware risk report for `qwen25_vl_72b`:

| Risk status | Count | Meaning |
| --- | ---: | --- |
| `probable_base_correction_no_support` | 16 | Qwen used the clean variant; no support provider preserved the student variant. |
| `probable_base_correction_with_preservation_support` | 12 | Qwen used the clean variant; at least one support provider preserved the student variant. |
| `base_lost_but_support_preserved` | 3 | Qwen lost the place; another provider preserved the student variant. |
| `base_preserved_student_mistake` | 3 | Qwen preserved the student variant. |
| `base_lost_no_signal` | 12 | No useful provider signal. |
| `base_lost_support_corrected` | 1 | Qwen lost the place; a support provider used the clean variant. |

The practical rule is to keep `qwen25_vl_72b` as the base transcription, but pass `probable_base_correction_with_preservation_support` and `base_lost_but_support_preserved` rows to the next LLM step as suspicious OCR-normalization points. On the current set, this gives 15 targeted places for review instead of manually reviewing all 47 student-vs-correct differences.

A strict mistake-preservation prompt was added for VLM-only experiments and completed successfully for `qwen25_vl_72b` and `gemma3_vision` on the 9-sample set. It reduced silent correction of student mistakes and improved normalized CER for both models.

Next research directions:

- expand the dataset to 30-50 checked Russian handwritten samples with writer/photo-condition labels;
- rerun only the strongest providers on the larger set first: `qwen25_vl_72b`, `gemma3_vision`, `yandex`, and optionally `cyrillic_trocr` as a line-level baseline;
- add page-quality metadata such as blur, tilt, contrast, grid type, and handwriting difficulty, then compare metrics by subgroup;
- improve line/region segmentation for hard pages before testing line-level OCR models again;
- test targeted second-pass VLM prompts only on low-confidence pages instead of rerunning every image;
- measure downstream preservation of student mistakes separately from raw OCR CER/WER.

Earlier one-sample Yandex baseline, kept as part of the research history:

| Provider | Dataset | CER | WER | Time | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `yandex` | first handwritten sample, original image | 0.056 | 0.171 | 2.912s | Strong OCR API baseline; preprocessed variant degraded to CER 0.065 and WER 0.257. |

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
