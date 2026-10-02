# MMSU-final

`mmsufinal` evaluates AudioMCQ's `MMSU.jsonl` with the existing `mmsu`
letter-answer protocol, extended to include the new E options. Select
`mmsufinal` through the usual lmms-eval or platform evaluation entry point.
No existing task or launcher is replaced.

## Data

- `dataset_kwargs.data_files.test` in `mmsufinal.yaml` selects the JSONL.
- `MMSUFINAL_DATASET_DIR` selects the local `mmsu` directory; its default is
  the dataset directory in this workspace. It must contain `data/*.parquet`;
  `_audio_cache/` is created on demand.
- Parquet contributes `sub_category` and `task_name`, joined by `id`, and
  embedded audio when the WAV cache needs to be generated.
  Questions, options and answers come from JSONL. Its `perception` and
  `reasoning` categories become the legacy `Perception` and `Reasoning` labels.
- Audio follows the original `mmsu` cache rules: replace non-`.wav` filename
  extensions with `.wav`, reuse nonempty files, and regenerate missing or
  empty files from the matching Parquet audio at 16 kHz mono. Cache writing
  reuses `mmsu._audio_cache_file`. Existing `.mp3`, `.flac` or `.ogg` cache
  names are not required; the source audio is embedded in `data/*.parquet`.
  An optional `lmms_eval_specific_kwargs.audio_cache_dir` overrides the cache
  location. Parquet audio is read only on cache misses, one row group at a
  time. Missing or unreadable source audio still fails the evaluation.

When moving the data, update the JSONL path in the task YAML and set
`MMSUFINAL_DATASET_DIR` to the corresponding dataset directory.

## Evaluation

The 4,996 four-option records use the existing MMSU prompt and scorer directly.
The four five-option records include E in both the prompt and parser's choice
list. They use the same legacy answer parser and single-letter comparison.
Generation remains maximum 256 output tokens, temperature 0, no sampling.
The `accuracy` metric and category/subcategory aggregation are inherited from
`mmsu` and return percentages, counting unparseable responses as wrong.

Options retain their order, empty strings and literal `"None"` values. The
answer text selects its first exact matching position. The three duplicated
`bow` answers at JSONL lines 4119, 4739 and 4742 therefore yield B, C and A,
matching the original Parquet labels. Another position with identical text
does not also count as correct. Invalid choice lists or missing answers raise
`ValueError` before indexing; the supported range is 2–26 options (A–Z).

The old parser's response-format behavior remains unchanged. This task does
not use the AudioMCQ answer-tag prompt, token-set scorer or sequence-text
scorer. The updated JSONL labels, including the four correct E answers, take
precedence over the old Parquet answers.
