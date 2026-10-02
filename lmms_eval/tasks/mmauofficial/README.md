# MMAU official test-mini and test

`mmauofficial` is now a group containing both splits. Previously this name ran
only mini; existing `--tasks mmauofficial` / `tasks: [mmauofficial]` configurations
now run 10,000 questions. Use `mmauofficial_test_mini` for the previous scope.

| Task | Questions | Local behavior | Accuracy |
| --- | ---: | --- | --- |
| `mmauofficial_test_mini` | 1,000 | Inference, official scoring, raw export | Available locally |
| `mmauofficial_test` | 9,000 | Inference, official Space submission export | Official service only |
| `mmauofficial` | 10,000 | Run both subtasks | Reported separately, never pooled |

The mini subtask uses:

- `/Users/andylubowen/Desktop/EIT-code/AudioMCQevaluator/mmau-test-mini.json`
- `/Users/andylubowen/Desktop/EIT-code/AudioMCQevaluator/test-mini-audios`

The audio directory ends in `audios`, not `audio`. The copy under
`/Users/andylubowen/Desktop/MMAU-v05.15.25/test-mini` contains the same mini
dataset in another format and is not loaded a second time.

The full-test subtask uses:

- `/Users/andylubowen/Desktop/MMAU-v05.15.25/test/test.parquet`
- `/Users/andylubowen/Desktop/MMAU-v05.15.25/test/test-audios`

All 1,000 mini answers are public (sound/music/speech: 333/334/333). All 9,000
public test answers are empty (3,000 questions per domain). Having the audio
files does not provide the hidden answers, so local test accuracy is unavailable.

## Official scoring equivalence

The mini scorer follows the original
[MMAU evaluation.py](https://github.com/Sakshi113/MMAU/blob/110127f54c0dfba3faa5ec9feee4a7e4148679c5/evaluation.py),
also available locally at
`/Users/andylubowen/Desktop/EIT-code/AudioMCQevaluator/evaluation.py`.
It uses the complete response in `model_output`, without extracting answer tags,
mapping letters to choices, truncating text, or performing semantic matching.
Overall accuracy is a percentage (0–100); task, difficulty, and sub-category
breakdowns use the official groups and denominators.

The official release specifies scoring, not a shared inference prompt. Matching
the scorer does not reproduce every leaderboard model's inference settings. This
adapter asks for the exact text of the selected option, without a letter,
explanation, or answer tags. It displays all options, including questions with
more than four choices. Reference answers and metadata are not model inputs.
Do not enable `reasoning_tags` or answer-extraction filters: those would remove
parts of the response before the official matcher sees it. The framework's
default first-response filter and outer-whitespace stripping do not change the
token-set score.

Both subtasks use the same prompt and generation settings.

`mmaunew2` already uses the same token-set rule on raw outputs for this mini
dataset. Its separate `model_prediction` field can contain an extracted or
letter-mapped answer. The mini subtask exports the original `model_output`, so
the saved prediction and the scored prediction are the same. Neither subtask
extracts answer tags or maps letters to options. `mmaunew2` is unchanged.

The dataset builder only converts the 13 single-element `category` lists into
strings so Arrow can load the mixed-type JSON column. This field is not used
by the official scorer; scoring text and grouping fields are unchanged.
Use the platform README's `datasets==3.6.0` environment: this task, like
`mmaunew2`, uses a local dataset builder script, which datasets 4.x no longer
loads. The test subtask uses the built-in Parquet loader directly.

## Official mini denominator behavior

An empty response string is present and counts as incorrect. If no response
exists (`results=[]`), the exported record has no `model_output` and is skipped,
as in the original script. If all outputs are missing, aggregation raises
`ZeroDivisionError`, also as in the original script. Its printed
`No prediction count: 0` does not detect skipped records. Verify the reported
sample count is 1,000 for a complete run; limited smoke runs report fewer.

## Full test: official submission, no local scoring

The official test service is
[MMAU Evaluation Space](https://huggingface.co/spaces/sonalkum/MMAU-Eval).
It reads `model_prediction`, whereas local `evaluation.py` reads `model_output`.
The test subtask never calls the local matcher with the empty public answers,
and has no local accuracy, score, or match metric. It exports a JSON array:

```json
[
  {"id": "test-sample-id", "model_prediction": "the original model response"}
]
```

Missing or empty responses retain their ID and an empty `model_prediction`;
the official service counts valid submitted IDs with empty predictions as
incorrect. A complete submission should contain 9,000 unique IDs. A limited
smoke run exports fewer records and must not be reported as full-test accuracy.
Files are generated locally and are not automatically uploaded.

## Run and independently rescore

An example Qwen2.5-Omni platform configuration is provided at
`configs/eval/lmms_eval/lmu/eval_aud_mmauofficial.yaml`. Like the neighboring
configurations, it uses `QWEN25_OMNI_MODEL_PATH` and `MLLM_COMP_EVAL_RUN_DIR`.
Run from the platform directory; the unchanged task selection now runs both:

```bash
python -m mllm_comp.cli.main eval \
  --config configs/eval/lmms_eval/lmu/eval_aud_mmauofficial.yaml
```

Append `--tasks mmauofficial_test_mini` or `--tasks mmauofficial_test` to run
one split. Direct lmms-eval invocation accepts the same task names:

```bash
--tasks mmauofficial                  # both splits
--tasks mmauofficial_test_mini        # mini only
--tasks mmauofficial_test             # full-test submission only
```

The mini subtask exports `submissions/mmauofficial_test_mini_results.json` under the
lmms-eval output directory. Recompute its scores with the unmodified official
script (replace the example output path):

```bash
python /Users/andylubowen/Desktop/EIT-code/AudioMCQevaluator/evaluation.py \
  --input /path/to/output/submissions/mmauofficial_test_mini_results.json
```

The exported `match` field is diagnostic; the official script recomputes it
from `answer`, `choices`, and `model_output`.

The test subtask separately exports `submissions/mmauofficial_test_submission.json`
under the same lmms-eval output directory. Upload that file to the official
Space for the hidden-answer score. Do not locally score the public test's empty
answers, and do not combine the two splits into a single 10,000-question accuracy.
