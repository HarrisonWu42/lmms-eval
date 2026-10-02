# MMAU bounded normalization and order correction

The task identifier is **`mmauoffical2`**, following the requested spelling.
The existing `mmauofficial` and `mmaunew2` tasks are unchanged. This version
reuses the `mmauofficial` prompt, audio functions, mini dataset loader and
generation settings. It is a local evaluation protocol, not an official
leaderboard reproduction.

| Task | Samples | Behavior |
| --- | ---: | --- |
| `mmauoffical2_test_mini` | 1,000 | Three local scores and normalized export |
| `mmauoffical2_test` | 9,000 | Normalized submission only; no local accuracy |
| `mmauoffical2` | 10,000 | Both splits, never pooled |

## Exact scope

1. Preserve responses that already match a literal choice, allowing case,
   whitespace and terminal periods for this recognition only. This protects
   musical notes and initials; the exported text itself is not rewritten.
2. Remove one leading valid option letter followed by a period, whitespace
   and a nonempty body, such as `A. Watching TV.` or `C. E`.
3. A bare letter without a literal-choice interpretation is invalid. A
   recognizable label/body disagreement is also invalid. Export an empty
   prediction and retain the ID, original response and reason in the audit.
   Never map a bare letter to an option or guess which conflicting part wins.
4. Keep both official token-set conditions, including the restriction on
   words unique to incorrect options. Do not remove explanations, courtesy,
   answer tags, or repair spelling/chord notation.
5. Add order checking only when two candidate choices have the same token
   set but different token sequences. Recognize complete candidate sequences
   at the start of the response, selecting the longest complete match to
   distinguish a short option from a longer option that begins the same way.
   Compare that sequence to the answer. Duplicate identical choices remain
   equivalent. Unresolved responses on these questions count as incorrect.

Parsing and submission normalization receive only the response and choices,
never the reference answer. Other tokenization behavior, including punctuation
handling, remains that of the official scorer. This does not implement general
semantic grading or resolve arbitrary contradictory explanations.

## Scores and files

Mini reports percentages on the same counted rows as `mmauofficial`:

- `mmauoffical2_accuracy`: normalized output plus the bounded order check.
- `mmauoffical2_official_raw_accuracy`: original received output, official rule.
- `mmauoffical2_submission_accuracy`: submitted text, official rule alone.

A missing mini response (`results=[]`) remains skipped, as in the original
task. A present empty string counts as incorrect. Full test always retains the
sample ID and uses an empty prediction for missing or invalid responses.

Files under the run's `submissions/` directory:

- `mmauoffical2_test_mini_results.json`: original mini records with normalized
  `model_output`; this is what the official mini script will score.
- `mmauoffical2_test_mini_audit.json`: raw and normalized outputs, all three
  scores and parsing diagnostics.
- `mmauoffical2_test_submission.json`: exactly `{id, model_prediction}` per
  test sample, containing normalized predictions without the removed labels.
- `mmauoffical2_test_audit.json`: original test responses and parsing diagnostics.

The audit's raw response is the string received by the task after any global
lmms-eval filters/stops. This task adds no reasoning-tag extraction and does
not change those framework settings. Actual answer letters such as the musical
note `D` are preserved: removing option labels does not mean deleting letters
that belong to the answer.

The official mini script recomputes the **submission accuracy**, not the order-
corrected accuracy. Likewise the [official full-test service](https://huggingface.co/spaces/sonalkum/MMAU-Eval/blob/main/app.py)
still uses token sets: normalized submissions fix label contamination, but
cannot repair its order/assignment ambiguity. Hidden-test accuracy cannot be
calculated or claimed locally. No files are uploaded automatically.

## Run

Use the same environment and dataset paths as `mmauofficial`, including
`datasets==3.6.0` for its shared mini loader. Run from the platform directory
with a full checkpoint or an already merged model:

```bash
CHECKPOINT_PATH="/path/to/current/checkpoint" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
EVAL_NUM_PROCESSES=4 \
bash scripts/eval/lmms_eval/lmu/start_eval_aud_mmauoffical2.sh
```

The Bash launcher selects `CHECKPOINT_PATH`, then `MODEL_PATH`, then an existing
`QWEN25_OMNI_MODEL_PATH`, in that order. It sets the model variable required by
the YAML; changing checkpoints does not require editing the config. It does
not merge LoRA adapters or change dataset paths.

Use `OUTPUT_DIR` to select the run directory. Otherwise it uses an existing
`MLLM_COMP_EVAL_RUN_DIR`, or a timestamped directory under `runs/lmu/eval/`.
An explicit output directory or `RUN_TAG` is reused as requested. The platform
launches Accelerate automatically. Additional CLI arguments are forwarded;
`--plan_only` prints the launch plan without loading the model.

Append `--tasks mmauoffical2_test_mini` or `--tasks mmauoffical2_test` to select
one split. Native lmms-eval also accepts `--tasks mmauoffical2`.
