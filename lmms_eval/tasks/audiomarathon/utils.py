import os
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger as eval_logger

TASK_DIR = Path(__file__).resolve().parent


def _candidate_roots():
    env_root = os.getenv("AUDIOMARATHON_ROOT", "").strip()
    if env_root:
        yield Path(env_root).expanduser()

    for parent in (Path.cwd(), *Path.cwd().parents, TASK_DIR, *TASK_DIR.parents):
        yield parent / "datasets" / "AudioMarathon"


def audiomarathon_root():
    for candidate in _candidate_roots():
        if candidate.exists():
            return candidate.resolve()
    env_root = os.getenv("AUDIOMARATHON_ROOT", "").strip()
    if env_root:
        return Path(env_root).expanduser().resolve()
    return (Path.cwd() / "datasets" / "AudioMarathon").resolve()


def resolve_audio_path(doc):
    audio_path = doc["audio_path"]
    path = Path(audio_path).expanduser()
    if path.is_absolute():
        return str(path)
    return str((audiomarathon_root() / audio_path).resolve())


def audiomarathon_doc_to_audio(doc):
    """Legacy/simple-model audio path: decode to array + sampling rate."""
    import soundfile as sf

    audio_path = resolve_audio_path(doc)
    audio, sampling_rate = sf.read(audio_path, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return [{"array": audio, "sampling_rate": sampling_rate}]


def _content_with_audio(doc, text):
    return [
        {
            "role": "user",
            "content": [
                {"type": "audio", "url": resolve_audio_path(doc)},
                {"type": "text", "text": text},
            ],
        }
    ]


def audiomarathon_doc_to_messages(doc, lmms_eval_specific_kwargs=None):
    return _content_with_audio(doc, audiomarathon_doc_to_text_mcq(doc, lmms_eval_specific_kwargs))


def audiomarathon_doc_to_messages_asr(doc, lmms_eval_specific_kwargs=None):
    return _content_with_audio(doc, audiomarathon_doc_to_text_asr(doc, lmms_eval_specific_kwargs))


def audiomarathon_doc_to_messages_summary(doc, lmms_eval_specific_kwargs=None):
    return _content_with_audio(doc, audiomarathon_doc_to_text_summary(doc, lmms_eval_specific_kwargs))


def _specific_kwargs(lmms_eval_specific_kwargs):
    return lmms_eval_specific_kwargs or {}


def audiomarathon_doc_to_text_mcq(doc, lmms_eval_specific_kwargs=None):
    kwargs = _specific_kwargs(lmms_eval_specific_kwargs)
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "Answer with only the option letter.")
    choices = doc.get("choices") or []
    return f"{pre_prompt}{doc['question']}\n" + "\n".join(choices) + f"\n{post_prompt}"


def audiomarathon_doc_to_text_asr(doc, lmms_eval_specific_kwargs=None):
    kwargs = _specific_kwargs(lmms_eval_specific_kwargs)
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    prompt = "Please transcribe the following audio. Only output the transcription."
    return f"{pre_prompt}{prompt}{post_prompt}"


def audiomarathon_doc_to_text_summary(doc, lmms_eval_specific_kwargs=None):
    kwargs = _specific_kwargs(lmms_eval_specific_kwargs)
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{doc['prompt']}{post_prompt}"


def _choice_letters(num_choices):
    return [chr(ord("A") + i) for i in range(num_choices)]


def _choice_body(choice):
    if not isinstance(choice, str):
        return str(choice)
    match = re.match(r"^\s*[A-Z]\s*[\.\)]\s*(.*)$", choice)
    return match.group(1).strip() if match else choice.strip()


def parse_multi_choice_response(response, choices):
    response = "" if response is None else str(response)
    letters = _choice_letters(len(choices))
    if not letters:
        return ""

    upper = response.upper()
    letter_set = "".join(re.escape(letter) for letter in letters)
    patterns = [
        rf"\(([{letter_set}])\)",
        rf"\bANSWER\s*[:：]?\s*([{letter_set}])\b",
        rf"\bOPTION\s*[:：]?\s*([{letter_set}])\b",
        rf"\b([{letter_set}])\s*[\.\)]",
        rf"\b([{letter_set}])\b",
    ]
    for pattern in patterns:
        matches = re.findall(pattern, upper)
        if matches:
            return matches[-1]

    lowered = response.lower()
    for letter, choice in zip(letters, choices):
        body = _choice_body(choice).lower()
        if body and body in lowered:
            return letter

    return ""


def audiomarathon_process_results_mcq(doc, results):
    response = results[0] if results else ""
    pred = parse_multi_choice_response(response, doc.get("choices") or [])
    gold = str(doc["answer"]).strip().upper()
    score = 1.0 if pred == gold else 0.0
    return {
        "audiomarathon_acc": {
            "score": score,
            "pred": pred,
            "gold": gold,
            "component": doc.get("component", ""),
            "task_name": doc.get("task_name", ""),
            "uniq_id": doc.get("uniq_id", ""),
        }
    }


def audiomarathon_aggregate_acc(results):
    if not results:
        return 0.0

    total = len(results)
    correct = sum(item["score"] for item in results)
    by_task = defaultdict(lambda: [0.0, 0])
    for item in results:
        key = item.get("task_name") or item.get("component") or "unknown"
        by_task[key][0] += item["score"]
        by_task[key][1] += 1

    eval_logger.info("=" * 60)
    eval_logger.info(f"AudioMarathon accuracy: {correct / total:.5f} ({int(correct)}/{total})")
    for key in sorted(by_task):
        task_correct, task_total = by_task[key]
        eval_logger.info(f"{key}: {task_correct / task_total:.5f} ({int(task_correct)}/{task_total})")
    eval_logger.info("=" * 60)
    return correct / total


def audiomarathon_process_results_asr(doc, results):
    return {
        "wer": {
            "gt": doc["transcript"],
            "pred": results[0] if results else "",
            "task": "asr_en",
        }
    }


def _lcs_len(a, b):
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for char_a in a:
        curr = [0]
        for idx_b, char_b in enumerate(b, 1):
            if char_a == char_b:
                curr.append(prev[idx_b - 1] + 1)
            else:
                curr.append(max(prev[idx_b], curr[-1]))
        prev = curr
    return prev[-1]


def _char_f1(reference, prediction):
    ref = re.sub(r"\s+", "", str(reference or ""))
    pred = re.sub(r"\s+", "", str(prediction or ""))
    if not ref and not pred:
        return 1.0
    if not ref or not pred:
        return 0.0
    overlap = _lcs_len(ref, pred)
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def audiomarathon_process_results_summary(doc, results):
    pred = results[0] if results else ""
    reference = doc["reference_summary"]
    return {
        "audiomarathon_summary_char_f1": {
            "score": _char_f1(reference, pred),
            "pred": pred,
            "reference_summary": reference,
            "meeting_id": doc.get("meeting_id", doc.get("uniq_id", "")),
        }
    }


def audiomarathon_aggregate_summary_char_f1(results):
    if not results:
        return 0.0
    score = sum(item["score"] for item in results) / len(results)
    eval_logger.info(f"AudioMarathon AliMeeting summary char-F1: {score:.5f}")
    return score
