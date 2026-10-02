# MMAR original accuracy

This task follows [MMAR's original answer-accuracy evaluator](https://github.com/ddlBoJack/MMAR/blob/main/code/evaluation.py), not the separate MMAR-Rubrics reasoning evaluator.

Version 2.0 reads the released `MMAR-meta.jsonl` directly. Version 1.0 derived answer text from the local Hugging Face Parquet mirror, which has five answer labels that differ from the official metadata. The Parquet mirror is no longer a source of task labels. Existing WAVs in `mmar/_audio_cache` are reused by the basename of the official `audio_path`, including multi-segment filenames.

Set both paths in `mmarofficial.yaml` for the execution host:

- `dataset_kwargs.data_files.test`: the official `MMAR-meta.jsonl`.
- `lmms_eval_specific_kwargs.default.audio_cache_dir`: the existing MMAR WAV cache.

Questions, choices, answers, and category metadata come from the official file. The prompt requests option text; the official repository does not specify a single common inference prompt for every model. Reference `thinking`, `cue`, and `rubric` fields are never included in the prompt. Empty padded choices are omitted from display but retained, unchanged, in the scorer input.

Predictions go into the original token-set rule without letter conversion, answer-tag extraction, or semantic matching. This deliberately preserves the original rule's limitations: it ignores word order and repetitions, and words outside the final answer may affect the score. Overall accuracy is a percentage; modality/category/sub-category breakdowns are logged. This task does not compute rubric scores or make judge API calls.

Select `mmarofficial` explicitly when launching evaluation. The project's existing batch scripts may still select `mmarnew`; changing this task does not change those scripts or historical results.
