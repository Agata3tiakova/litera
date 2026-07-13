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

- 14 Russian handwritten school-text samples;
- all 14 samples contain the same base text, written by different students with different handwriting styles and different mistakes;
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

Other VLM providers are integrated in the codebase, but the current benchmark focuses on models that completed reliable runs.

## Main OCR Metrics

Current 14-sample benchmark, using `student_text` as the primary reference:

| Provider | Type | Runs | Avg normalized CER | Avg normalized WER | Avg clean normalized CER | Avg clean normalized WER | Avg time |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen25_vl_72b` | VLM | 14 | 0.097 | 0.217 | 0.123 | 0.220 | 6.200s |
| `gemma3_vision` | VLM | 14 | 0.197 | 0.372 | 0.212 | 0.359 | 4.992s |
| `yandex` | OCR API | 14 | 0.310 | 0.480 | 0.349 | 0.520 | 2.499s |
| `cyrillic_trocr` | Transformer OCR | 9/14 | 0.686 | 0.958 | 0.705 | 0.996 | 296.509s |
| `easyocr` | OCR | 14 | 0.868 | 1.224 | 0.854 | 1.216 | 12.937s |
| `tesseract` | OCR | 14 | 0.927 | 1.349 | 0.922 | 1.335 | 2.161s |
| `trocr` | Transformer OCR | 14 | 0.984 | 1.000 | 0.985 | 1.000 | 5.801s |

The ranking did not change on the expanded dataset: Qwen remains first, Gemma second, and Yandex the strongest classic OCR/API baseline.

`cyrillic_trocr` completed only 9 of 14 samples on the current environment. On 5 of 14 samples it did not finish: PyTorch is installed as CPU-only (`torch 2.12.1+cpu`), and the Cyrillic TrOCR line-level provider timed out after 15 minutes for the remaining batch and after 5 minutes for a single sample from that group.

### Interpretation

`qwen25_vl_72b` is the strongest model by transcription quality. `gemma3_vision` is second. `yandex` is the strongest classic OCR/API baseline and is faster than the VLM providers, but it is significantly worse on difficult Russian handwriting. Classic local OCR models (`tesseract`, `easyocr`, generic `trocr`) are useful as baselines but are not competitive for this task.

`cyrillic_trocr` improves over generic TrOCR, but it is slow on CPU and often produces Church Slavonic-like tokens. It should be treated as a line-level Cyrillic OCR baseline, not as the leading approach for full-page Russian school notebooks.

## Strict Prompt Experiment

Because VLMs tend to correct text by context, a stricter prompt was added:

`preserve_student_errors_strict`

This prompt explicitly asks the model to preserve spelling mistakes, wrong word forms, punctuation mistakes, line breaks, and visually written words even when they are grammatically wrong.

| Provider | Prompt | Runs | Avg Norm CER | Avg Norm WER | Preserved mistakes | Corrected mistakes | Preservation rate | Correction rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `qwen25_vl_72b` | default | 14 | 0.097 | 0.217 | 8 | 49 | 0.103 | 0.628 |
| `qwen25_vl_72b` | `preserve_student_errors_strict` | 14 | 0.088 | 0.253 | 21 | 28 | 0.269 | 0.359 |
| `gemma3_vision` | default | 14 | 0.197 | 0.372 | 5 | 44 | 0.064 | 0.564 |
| `gemma3_vision` | `preserve_student_errors_strict` | 14 | 0.152 | 0.325 | 14 | 25 | 0.179 | 0.321 |

### Strict Prompt Findings

The strict prompt improved mistake preservation for both VLM providers. It also improved normalized CER for both models. Qwen's normalized WER rose compared with the default prompt, but the overall result is better for the educational OCR task because fewer real student mistakes are silently corrected. Gemma improved both CER and WER with the strict prompt.

For this project, `preserve_student_errors_strict` should be treated as the preferred VLM OCR prompt.

## Student Mistake Preservation

The 14-sample dataset contains 78 detected student-vs-correct differences. These are real places where the student wrote something different from the clean reference.

| Provider | Checked mistakes | Preserved | Corrected | Lost | Preservation rate | Correction rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `yandex` | 78 | 28 | 18 | 32 | 0.359 | 0.231 |
| `cyrillic_trocr` | 47 | 9 | 0 | 38 | 0.191 | 0.000 |
| `qwen25_vl_72b` | 78 | 8 | 49 | 21 | 0.103 | 0.628 |
| `gemma3_vision` | 78 | 5 | 44 | 29 | 0.064 | 0.564 |
| `tesseract` | 78 | 2 | 0 | 76 | 0.026 | 0.000 |
| `easyocr` | 78 | 1 | 0 | 77 | 0.013 | 0.000 |
| `trocr` | 78 | 1 | 0 | 77 | 0.013 | 0.000 |

`cyrillic_trocr` preservation metrics cover only the 9 samples it completed; the remaining 5 of 14 samples timed out.

### Preservation Findings

This table shows a key research tradeoff. Qwen is best by CER/WER, but the default prompt often normalizes mistakes to the clean reference. Yandex preserves more student variants, but also loses many difficult words. Cyrillic TrOCR sometimes preserves student-like variants, but loses too much text overall.

For educational analysis, the best approach is not simply "use the lowest CER output". The OCR layer should also flag places where the model may have corrected the student's mistake.

## Preservation-Aware Risk Report

The project includes a risk report that uses Qwen as the base transcription and checks whether other providers preserve student variants that Qwen corrected or lost. The current risk table covers all 78 detected student-vs-correct differences in the 14-sample dataset.

| Risk status | Count | Meaning |
| --- | ---: | --- |
| `probable_base_correction_with_preservation_support` | 26 | Qwen used the clean variant; at least one support provider preserved the student variant. |
| `probable_base_correction_no_support` | 23 | Qwen used the clean variant; no support provider preserved the student variant. |
| `base_lost_no_signal` | 15 | No useful provider signal. |
| `base_preserved_student_mistake` | 8 | Qwen preserved the student variant. |
| `base_lost_but_support_preserved` | 3 | Qwen lost the place; another provider preserved the student variant. |
| `base_lost_support_corrected` | 3 | Qwen lost the place; a support provider used the clean variant. |

The practical rule is to keep Qwen as the base transcription, but pass `probable_base_correction_with_preservation_support` and `base_lost_but_support_preserved` rows to the next LLM step as suspicious OCR-normalization points. On the 14-sample run, this gives 29 targeted places for review instead of manually reviewing all 78 differences.

Preservation support providers in this report:

| Provider | Preserved student variant count |
| --- | ---: |
| `yandex` | 28 |
| `cyrillic_trocr` | 9 |
| `gemma3_vision` | 5 |

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

1. Classic OCR and VLMs make different kinds of errors because they solve different tasks. OCR systems mostly recognize visual symbols and local word shapes, so they often lose handwritten words, split words incorrectly, or produce noisy fragments on notebook photos. VLMs read the image together with language context, so they produce more coherent text and lower CER/WER, but they can replace the student's misspelled word with the grammatically expected word.
2. This structural difference explains the metric gap. VLMs have much better transcription metrics on the 14-sample Russian handwriting benchmark: Qwen2.5-VL-72B reaches normalized CER 0.097 and WER 0.217, while Yandex, the strongest OCR/API baseline, reaches CER 0.310 and WER 0.480. Local classic OCR baselines are much weaker on full-page handwriting.
3. The same structural difference creates a separate educational risk. A VLM can look better by CER/WER while silently correcting real student mistakes. Default Qwen preserved only 8 of 78 checked student mistakes and corrected 49, so raw transcription quality alone is not enough for this project.
4. The strict VLM prompt was added to reduce contextual normalization. For Qwen, it improved normalized CER from 0.097 to 0.088 and raised mistake preservation from 8 to 21 cases, while reducing corrections from 49 to 28. For Gemma, it improved normalized CER from 0.197 to 0.152 and WER from 0.372 to 0.325, with better mistake preservation.
5. The benchmark was changed to use `student_text` as the primary reference and `correct_text` only as a secondary clean reference. This prevents models from being rewarded for correcting the student's writing when the actual OCR task is to transcribe what was written.
6. A preservation-aware risk report was added because no single model is reliable enough by itself. Qwen remains the base transcription model, but Yandex, Cyrillic TrOCR, and Gemma are used as support signals when they preserve a student variant that Qwen corrected or lost. On the 14-sample run, this identifies 29 targeted suspicious places instead of manually reviewing all 78 differences.
7. The practical improvement is a two-layer OCR workflow: first choose the strongest transcription model by CER/WER, then add prompt constraints and cross-model preservation checks to protect real student mistakes before the separate LLM grammar-analysis step.

## Next Steps

- expand the dataset to 30-50 checked Russian handwritten samples;
- add metadata for blur, tilt, contrast, grid type, and handwriting difficulty;
- evaluate strict VLM prompts on the larger set;
- pass preservation-risk rows into the LLM analysis stage;
- test whether LLM feedback improves when it receives suspicious OCR-normalization points;
- explore better line/region segmentation for line-level OCR models.
