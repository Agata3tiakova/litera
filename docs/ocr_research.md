# OCR Research: Russian Handwritten School Text

This document describes the OCR/VLM benchmark built for the Litera school project.

Litera started as a school-oriented web application for recognizing handwritten student text and analyzing writing quality with AI. During development it became clear that Russian handwritten OCR is a separate hard problem: classic OCR systems often fail on notebook photos, while multimodal vision-language models can read more context but may silently correct student mistakes. Because of that, the project includes a benchmark and research workflow for comparing OCR providers before using their output in educational feedback.

## Research Goal

The goal is not only to get the cleanest transcription. For an educational task, the system must preserve what the student actually wrote, including spelling mistakes. A model that produces lower CER/WER can still be harmful if it silently fixes student errors before the grammar-analysis step.

The benchmark therefore tracks two layers:

- OCR/VLM transcription quality: CER, WER, speed, and cost;
- student mistake preservation: whether OCR keeps the student's real mistake, corrects it, or loses the word.

## Dataset

Current benchmark dataset:

- 9 Russian handwritten school-text samples;
- phone photos and notebook/grid-paper images;
- manually checked `student_text` references that preserve student mistakes;
- separate `correct_text` references with the clean version of the text.

The primary OCR metric uses `student_text`, not `correct_text`, because OCR should transcribe the student's real writing.

## Models Compared

Classic OCR and OCR APIs:

- `tesseract`
- `easyocr`
- `yandex`

Transformer OCR:

- `trocr`
- `cyrillic_trocr`

Vision-language models:

- `qwen25_vl_72b`
- `gemma3_vision`

Other VLM providers are integrated in the codebase, but the current 9-sample benchmark focuses on models that completed reliable runs.

## Main OCR Metrics

Current 9-sample benchmark, using `student_text` as the primary reference:

| Provider | Type | Avg normalized CER | Avg normalized WER | Avg time |
| --- | --- | ---: | ---: | ---: |
| `qwen25_vl_72b` | VLM | 0.133 | 0.277 | 6.129s |
| `gemma3_vision` | VLM | 0.271 | 0.484 | 4.006s |
| `yandex` | OCR API | 0.417 | 0.609 | 2.674s |
| `cyrillic_trocr` | Transformer OCR | 0.686 | 0.958 | 296.509s |
| `easyocr` | OCR | 0.883 | 1.133 | 17.974s |
| `tesseract` | OCR | 0.962 | 1.455 | 3.013s |
| `trocr` | Transformer OCR | 0.978 | 1.000 | 5.780s |

### Interpretation

`qwen25_vl_72b` is the strongest model by transcription quality. `gemma3_vision` is second. `yandex` is the strongest classic OCR/API baseline and is faster than the VLM providers, but it is significantly worse on difficult Russian handwriting. Classic local OCR models (`tesseract`, `easyocr`, generic `trocr`) are useful as baselines but are not competitive for this task.

`cyrillic_trocr` improves over generic TrOCR, but it is slow on CPU and often produces Church Slavonic-like tokens. It should be treated as a line-level Cyrillic OCR baseline, not as the leading approach for full-page Russian school notebooks.

## Strict Prompt Experiment

Because VLMs tend to correct text by context, a stricter prompt was added:

`preserve_student_errors_strict`

This prompt explicitly asks the model to preserve spelling mistakes, wrong word forms, punctuation mistakes, line breaks, and visually written words even when they are grammatically wrong.

| Provider | Prompt | Avg Norm CER | Avg Norm WER | Preserved mistakes | Corrected mistakes | Preservation rate | Correction rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen25_vl_72b` | default | 0.133 | 0.277 | 3 | 28 | 0.064 | 0.596 |
| `qwen25_vl_72b` | `preserve_student_errors_strict` | 0.109 | 0.290 | 8 | 20 | 0.170 | 0.426 |
| `gemma3_vision` | default | 0.271 | 0.484 | 2 | 21 | 0.043 | 0.447 |
| `gemma3_vision` | `preserve_student_errors_strict` | 0.207 | 0.391 | 8 | 14 | 0.170 | 0.298 |

### Strict Prompt Findings

The strict prompt improved mistake preservation for both VLM providers. It also improved normalized CER for both models. Qwen's normalized WER rose slightly, but the overall result is better for the educational OCR task because fewer real student mistakes are silently corrected.

For this project, `preserve_student_errors_strict` should be treated as the preferred VLM OCR prompt.

## Student Mistake Preservation

The dataset contains 47 detected student-vs-correct differences. These are real places where the student wrote something different from the clean reference.

| Provider | Checked mistakes | Preserved | Corrected | Lost | Preservation rate | Correction rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yandex` | 47 | 12 | 10 | 25 | 0.255 | 0.213 |
| `cyrillic_trocr` | 47 | 9 | 0 | 38 | 0.191 | 0.000 |
| `qwen25_vl_72b` | 47 | 3 | 28 | 16 | 0.064 | 0.596 |
| `gemma3_vision` | 47 | 2 | 21 | 24 | 0.043 | 0.447 |
| `tesseract` | 47 | 1 | 0 | 46 | 0.021 | 0.000 |
| `easyocr` | 47 | 0 | 0 | 47 | 0.000 | 0.000 |
| `trocr` | 47 | 0 | 0 | 47 | 0.000 | 0.000 |

### Preservation Findings

This table shows a key research tradeoff. Qwen is best by CER/WER, but the default prompt often normalizes mistakes to the clean reference. Yandex preserves more student variants, but also loses many difficult words. Cyrillic TrOCR sometimes preserves student-like variants, but loses too much text overall.

For educational analysis, the best approach is not simply "use the lowest CER output". The OCR layer should also flag places where the model may have corrected the student's mistake.

## Preservation-Aware Risk Report

The project includes a risk report that uses Qwen as the base transcription and checks whether other providers preserve student variants that Qwen corrected or lost.

| Risk status | Count | Meaning |
| --- | ---: | --- |
| `probable_base_correction_no_support` | 16 | Qwen used the clean variant; no support provider preserved the student variant. |
| `probable_base_correction_with_preservation_support` | 12 | Qwen used the clean variant; at least one support provider preserved the student variant. |
| `base_lost_but_support_preserved` | 3 | Qwen lost the place; another provider preserved the student variant. |
| `base_preserved_student_mistake` | 3 | Qwen preserved the student variant. |
| `base_lost_no_signal` | 12 | No useful provider signal. |
| `base_lost_support_corrected` | 1 | Qwen lost the place; a support provider used the clean variant. |

The practical rule is to keep Qwen as the base transcription, but pass `probable_base_correction_with_preservation_support` and `base_lost_but_support_preserved` rows to the next LLM step as suspicious OCR-normalization points. On the current set, this gives 15 targeted places for review instead of manually reviewing all 47 differences.

## Typical Error Patterns

### Qwen2.5-VL

Strengths:

- best overall CER/WER;
- strong contextual reading;
- handles difficult photos better than classic OCR.

Weaknesses:

- silently corrects student spelling mistakes;
- may normalize nonstandard words to standard Russian;
- one difficult page can dominate the error rate.

Typical errors:

- `суботу` -> `субботу`;
- `облока` -> `облака`;
- `вернутся` -> `вернуться`;
- `Черес` -> `Через`.

### Gemma 3 Vision

Strengths:

- second-best VLM baseline;
- strict prompt improves both CER and mistake preservation;
- fewer severe failures than classic OCR.

Weaknesses:

- still corrects many student mistakes;
- sometimes loses words in difficult handwriting;
- less accurate than Qwen on the same set.

Typical errors:

- normalizes spelling mistakes;
- misses words on dense or messy lines;
- confuses visually close handwritten forms.

### Yandex Vision

Strengths:

- strongest classic OCR/API baseline;
- fast;
- preserves student variants more often than default VLM prompts.

Weaknesses:

- loses many difficult handwritten words;
- weaker than VLMs on full-page noisy handwriting;
- performance varies strongly by image quality.

Typical errors:

- partial word recognition;
- substitutions on hard handwriting;
- missing words in dense lines.

### Cyrillic TrOCR

Strengths:

- better than generic TrOCR;
- useful as a Cyrillic line-level OCR baseline;
- occasionally preserves student-like spellings.

Weaknesses:

- very slow on CPU;
- not robust for full notebook pages;
- often outputs Church Slavonic-like tokens.

Typical errors:

- hallucinated archaic/Cyrillic tokens;
- many extra words;
- missing full lines or large parts of text.

### Tesseract and EasyOCR

Strengths:

- useful local baselines;
- no paid API dependency.

Weaknesses:

- poor Russian handwritten recognition;
- many missing and extra words;
- high WER on notebook photos.

Typical errors:

- Latin/Cyrillic confusion;
- fragmented words;
- missing handwritten lines;
- punctuation and layout noise.

## Conclusions

1. VLMs are currently the strongest direction for Russian handwritten school text.
2. Qwen2.5-VL-72B is the best base transcription model by CER/WER.
3. The strict mistake-preservation prompt improves VLM usefulness for educational analysis.
4. Classic OCR systems are useful baselines, but not strong enough for the main handwritten recognition task.
5. Yandex remains a valuable OCR API baseline because it is fast and sometimes preserves student mistakes better than VLMs.
6. OCR evaluation must use `student_text` as the primary reference; clean corrected text should be secondary.
7. For educational feedback, raw CER/WER is not enough. The system must measure whether OCR preserved student mistakes.

## Next Steps

- expand the dataset to 30-50 checked Russian handwritten samples;
- add metadata for blur, tilt, contrast, grid type, and handwriting difficulty;
- evaluate strict VLM prompts on the larger set;
- pass preservation-risk rows into the LLM analysis stage;
- test whether LLM feedback improves when it receives suspicious OCR-normalization points;
- explore better line/region segmentation for line-level OCR models.
