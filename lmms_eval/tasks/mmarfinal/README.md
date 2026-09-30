# MMAR-final

`mmarfinal` evaluates AudioMCQ's `MMAR.jsonl` with the existing `mmar`
letter-answer protocol. Select `mmarfinal` through the usual lmms-eval or
platform evaluation entry point. No existing task or launcher is replaced.

## Data

- `dataset_kwargs.data_files.test` in `mmarfinal.yaml` selects the JSONL.
- `MMARFINAL_DATASET_DIR` selects the local `mmar` directory; its default is
  the dataset directory in this workspace. It must contain `data/*.parquet`
  and `_audio_cache/`.
- Parquet contributes only `modality` and `subcategory`, joined by `id`.
  Questions, options, answers and categories come from JSONL.
- Audio uses `_audio_cache/<basename of audio_path>`. An optional
  `lmms_eval_specific_kwargs.audio_cache_dir` overrides the cache location.
  Missing or empty audio raises an error. Audio is neither decoded from
  Parquet nor regenerated during evaluation.

When moving the data, update the JSONL path in the task YAML and set
`MMARFINAL_DATASET_DIR` to the corresponding dataset directory.

## Evaluation

Options retain their order and content, including empty strings. Their
positions become A, B, C, etc.; the JSONL answer text selects the first exact
matching position. The supplied 1,000-row MMAR release has unique answer
matches. Invalid choice lists or missing answers raise `ValueError` before
indexing; the supported label range is A–Z (2–26 options).

The prompt, answer parser, result schema, `mmar_accuracy` aggregation and
generation settings are inherited from `mmar`: maximum 128 output tokens,
temperature 0, no sampling. Accuracy is the percentage of samples whose
parsed letter equals the target letter; unparseable responses count as wrong.

For protocol parity, the legacy prompt still says A–D even on the existing
six-option MMAR record. All six options remain visible and the parser accepts
their actual labels. Empty-option and response-format behavior is also
inherited unchanged. This task does not use AudioMCQ's answer-tag prompt or
token-set scorer.

The JSONL contains updated labels and text, so matching the old scoring
algorithm does not imply matching scores obtained on the old Parquet labels.
