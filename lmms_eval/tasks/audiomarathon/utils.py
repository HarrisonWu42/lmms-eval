import os
import re
from collections import defaultdict
from pathlib import Path

from loguru import logger as eval_logger

TASK_DIR = Path(__file__).resolve().parent
MANIFEST_DIR = TASK_DIR / "manifests"
LIBRISPEECH_LONG_MANIFEST = str(MANIFEST_DIR / "librispeech_long.jsonl")
RACE_AUDIO_MANIFEST = str(MANIFEST_DIR / "race_audio.jsonl")
HAD_MANIFEST = str(MANIFEST_DIR / "had.jsonl")
GTZAN_MANIFEST = str(MANIFEST_DIR / "gtzan.jsonl")
TAU_MANIFEST = str(MANIFEST_DIR / "tau.jsonl")
VESUS_MANIFEST = str(MANIFEST_DIR / "vesus.jsonl")
SLUE_MANIFEST = str(MANIFEST_DIR / "slue.jsonl")
DESED_MANIFEST = str(MANIFEST_DIR / "desed.jsonl")
VOXCELEB_GENDER_MANIFEST = str(MANIFEST_DIR / "voxceleb_gender.jsonl")
VOXCELEB_AGE_MANIFEST = str(MANIFEST_DIR / "voxceleb_age.jsonl")

QWEN_SYSTEM_PROMPT = (
    "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, "
    "capable of perceiving auditory and visual inputs, as well as generating text and speech."
)


def official_qwen_system_prompt(doc):
    component = doc.get("component", "")
    if component == "DESED":
        return (
            "You are a helpful assistant that analyzes audio to detect and classify sound events. "
            "Please listen carefully and select the most appropriate answer from the given choices."
        )
    if component == "HAD":
        return "You are a helpful assistant that analyzes audio to determine authenticity."
    if component == "GTZAN":
        return f"{QWEN_SYSTEM_PROMPT} You are a helpful audio analysis assistant."
    return QWEN_SYSTEM_PROMPT


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
    return [
        {
            "array": audio,
            "sampling_rate": sampling_rate,
            "system_prompt": official_qwen_system_prompt(doc),
        }
    ]


def _content_with_audio(doc, text):
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": official_qwen_system_prompt(doc)}],
        },
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


def _specific_kwargs(lmms_eval_specific_kwargs):
    return lmms_eval_specific_kwargs or {}


def _choice_map(doc):
    choices = doc.get("choices") or []
    return {chr(ord("A") + index): _choice_body(choice) for index, choice in enumerate(choices)}


def _format_choices(choices, separator=". "):
    return "\n".join(f"{letter}{separator}{text}" for letter, text in choices.items())


def official_qwen_mcq_prompt(doc):
    component = doc.get("component", "")
    question = doc["question"]
    choices = _choice_map(doc)

    if component == "RACE-audio":
        task_instruction = (
            "You are a helpful assistant that analyzes reading comprehension passages with audio narration. "
            "Please listen to the passage and answer the multiple-choice question based on what you heard."
        )
        instruction = (
            "Listen to this audio of a passage being read aloud, then answer the multiple-choice "
            "question based solely on the information from the audio."
        )
        return (
            f"{task_instruction}\n\n{instruction}\n\nQuestion: {question}\n\nOptions:\n"
            f"{_format_choices(choices)}\n\nRespond with only the letter of the correct option "
            "(A, B, C, or D)."
        )

    if component == "HAD":
        return (
            "Listen to this audio clip carefully. Is this audio completely authentic (real) or does it "
            "contain any artificially synthesized segments (fake)? If it is completely real, answer 'a'. "
            "If it contains any fake segments, answer 'b'. Answer with only 'a' or 'b'."
        )

    if component == "GTZAN":
        return (
            "Listen to this audio segment and identify the music genre based on what you hear.\n\n"
            f"Question: {question}\n\nOptions:\n{_format_choices(choices)}\n\n"
            "Respond with only the letter of the correct option (A, B, C, or D)."
        )

    if component == "TAU":
        task_instruction = (
            "You are a helpful assistant that analyzes urban soundscape audio to identify acoustic scenes. "
            "Please listen to the audio carefully and classify the scene type."
        )
        return (
            f"{task_instruction}\n\nListen to this audio and identify the acoustic scene. "
            f"Choose the most appropriate option.\n{_format_choices(choices, ': ')}\n"
            "Respond with only the letter of your answer (A, B, C, or D)."
        )

    if component == "VESUS":
        task_instruction = (
            "You are a helpful assistant that analyzes speech audio to recognize emotions. "
            "Please listen to the voice carefully and identify the emotional state of the speaker."
        )
        return (
            f"{task_instruction}\n\n{question}\n\n{_format_choices(choices, ') ')}\n\n"
            "Please select the correct answer (A, B, C, or D)."
        )

    if component == "SLUE":
        task_instruction = (
            "You are a helpful assistant that analyzes speech audio for named entity recognition. "
            "Please listen carefully and extract the requested named entities from the speech."
        )
        return (
            f"{task_instruction}\n\n{question}\n\n{_format_choices(choices)}\n\n"
            "Please listen to the audio and select the correct answer. Reply with only the letter "
            "(A, B, C, or D)."
        )

    if component == "DESED":
        return (
            f"{question}\n\n{_format_choices(choices)}\n\nPlease listen to the audio and select "
            "the correct answer. Reply with only the letter (A, B, C, or D)."
        )

    if doc.get("task_name") == "Speaker gender classification":
        task_instruction = (
            "You are a helpful assistant that analyzes speech audio to classify speaker gender. "
            "Please listen to the voice carefully and determine the speaker's gender."
        )
        instruction = (
            "Listen to this audio and identify the speaker's gender. Is this a male or female voice? "
            "If it is a male, answer 'a'. If it is a female, answer 'b'. Answer with only 'a' or 'b'."
        )
        return f"{task_instruction}\n\n{instruction}"

    if doc.get("task_name") == "Speaker age classification":
        task_instruction = (
            "You are a helpful assistant that analyzes speech audio to estimate speaker age. "
            "Please listen to the voice carefully and classify the speaker's age group."
        )
        return (
            f"{task_instruction}\n\n{question}\n\n{_format_choices(choices, ') ')}\n\n"
            "Please select the correct answer (A, B, C, D, or E)."
        )

    raise ValueError(f"Unsupported AudioMarathon component/task: {component!r}/{doc.get('task_name')!r}")


def audiomarathon_doc_to_text_mcq(doc, lmms_eval_specific_kwargs=None):
    kwargs = _specific_kwargs(lmms_eval_specific_kwargs)
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    return f"{pre_prompt}{official_qwen_mcq_prompt(doc)}{post_prompt}"


def audiomarathon_doc_to_text_asr(doc, lmms_eval_specific_kwargs=None):
    kwargs = _specific_kwargs(lmms_eval_specific_kwargs)
    pre_prompt = kwargs.get("pre_prompt", "")
    post_prompt = kwargs.get("post_prompt", "")
    prompt = (
        "You are a helpful assistant that transcribes speech audio. Please listen carefully and provide "
        "the exact transcription of what is spoken in the audio.\n\nTranscribe this audio accurately. "
        "Remove hesitation words like 'um', 'uh'. Your response should be formatted as follows: "
        "Spoken Content: <transcribed text here>"
    )
    return f"{pre_prompt}{prompt}{post_prompt}"


def _choice_letters(num_choices):
    return [chr(ord("A") + i) for i in range(num_choices)]


def _choice_body(choice):
    if not isinstance(choice, str):
        return str(choice)
    match = re.match(r"^\s*[A-Z]\s*[\.\)]\s*(.*)$", choice)
    return match.group(1).strip() if match else choice.strip()


def _strip_assistant_prefix(response):
    response = "" if response is None else str(response)
    if "assistant\n" in response:
        index = response.rfind("assistant\n") + len("assistant\n")
        response = response[index:]
    return response.strip()


def _letter_for_unique_choice_text(response, choices):
    lowered = response.lower()
    matches = []
    for letter, choice in zip(_choice_letters(len(choices)), choices):
        body = _choice_body(choice).lower()
        if body and body in lowered:
            matches.append(letter)
    return matches[0] if len(matches) == 1 else ""


def parse_multi_choice_response(response, choices, component="", task_name=""):
    letters = _choice_letters(len(choices))
    if not letters:
        return ""

    response = "" if response is None else str(response)

    # The official RACE helper returns the first A-D character in the reply.
    if component == "RACE-audio":
        return next((char for char in response.strip().upper() if char in letters), "")

    if component in {"GTZAN", "DESED"}:
        response = _strip_assistant_prefix(response)
    else:
        response = response.strip()

    upper = response.upper()
    lowered = response.lower().strip()

    if component == "HAD":
        for letter in letters[:2]:
            low_letter = letter.lower()
            if (
                lowered == low_letter
                or lowered.startswith(f"{low_letter}.")
                or lowered.startswith(f"{low_letter})")
                or lowered.endswith(f" {low_letter}")
            ):
                return letter
        for letter in letters[:2]:
            low_letter = letter.lower()
            if (
                f"option {low_letter}" in lowered
                or f"choice {low_letter}" in lowered
                or f"{low_letter})" in lowered
                or f" {low_letter} " in lowered
            ):
                return letter
        semantic = _letter_for_unique_choice_text(response, choices)
        if semantic:
            return semantic
        real = re.search(r"\breal\b|\bauthentic\b|\bgenuine\b|\bcompletely authentic\b", lowered)
        fake = re.search(r"\bfake\b|\bartificial\b|\bsynthetic\b|\bsynthesized\b|\bcontains.*fake\b", lowered)
        if real and not fake:
            return "A"
        if fake and not real:
            return "B"
        return ""

    if task_name == "Speaker gender classification":
        for letter in letters[:2]:
            low_letter = letter.lower()
            if lowered == low_letter or lowered.startswith(f"{low_letter}.") or lowered.startswith(f"{low_letter})"):
                return letter
        for letter in letters[:2]:
            low_letter = letter.lower()
            if f"option {low_letter}" in lowered or f"choice {low_letter}" in lowered or f"{low_letter})" in lowered:
                return letter
        semantic = _letter_for_unique_choice_text(response, choices)
        if semantic:
            return semantic
        male = re.search(r"\bmale\b", lowered)
        female = re.search(r"\bfemale\b", lowered)
        if male and not female:
            return next((letter for letter, choice in zip(letters, choices) if _choice_body(choice).lower() == "male"), "")
        if female and not male:
            return next((letter for letter, choice in zip(letters, choices) if _choice_body(choice).lower() == "female"), "")
        return ""

    if task_name == "Speaker age classification":
        for letter in letters:
            low_letter = letter.lower()
            if (
                lowered == low_letter
                or lowered.startswith(f"{low_letter}.")
                or lowered.startswith(f"{low_letter})")
                or lowered.endswith(f" {low_letter}")
            ):
                return letter
        present = [letter for letter in letters if re.search(rf"\b{letter}\b", upper)]
        if len(present) == 1:
            return present[0]
        for letter in letters:
            low_letter = letter.lower()
            if f"option {low_letter}" in lowered or f"choice {low_letter}" in lowered or f"answer {low_letter}" in lowered:
                return letter
        return _letter_for_unique_choice_text(response, choices)

    if component == "GTZAN":
        if upper in letters:
            return upper
        for letter in letters:
            if upper.startswith(f"{letter}.") or upper.startswith(f"{letter})") or upper.endswith(f" {letter}"):
                return letter
        for letter in letters:
            low_letter = letter.lower()
            if f"option {low_letter}" in lowered or f"choice {low_letter}" in lowered or f"{low_letter})" in lowered:
                return letter
        match = re.search(r"\b([ABCD])\b", upper)
        if match:
            return match.group(1)
        match = re.search(r"[(\[]?([ABCD])[)\].]?", upper)
        return match.group(1) if match else ""

    if component == "DESED":
        if upper in letters:
            return upper
        match = re.search(r"\b([ABCD])\b", upper)
        if match:
            return match.group(1)
        match = re.search(r"[(\[]?([ABCD])[)\].]?", upper)
        return match.group(1) if match else ""

    if component == "SLUE":
        if upper in letters:
            return upper
        for letter in letters:
            if upper.startswith(letter) and len(upper) <= 3:
                return letter
        match = re.search(r"\b([ABCD])\b", upper)
        if match:
            return match.group(1)
        match = re.search(r"[(\[]?([ABCD])[)\].]?", upper)
        if match:
            return match.group(1)
        match = re.search(r"(?:OPTION|CHOICE)\s+([ABCD])", upper)
        return match.group(1) if match else ""

    if component == "TAU":
        if upper in letters:
            return upper
        for letter in letters:
            if any(lowered.startswith(f"{letter.lower()}{suffix}") for suffix in (".", ")", ":")):
                return letter
        for letter in letters:
            low_letter = letter.lower()
            if f"option {low_letter}" in lowered or f"choice {low_letter}" in lowered or f"{low_letter})" in lowered:
                return letter
        semantic = _letter_for_unique_choice_text(response, choices)
        if semantic:
            return semantic
        best_letter = ""
        best_overlap = 0
        for letter, choice in zip(letters, choices):
            keywords = _choice_body(choice).lower().split(" - ")[0].split()
            overlap = sum(keyword in lowered for keyword in keywords)
            if overlap > best_overlap:
                best_letter = letter
                best_overlap = overlap
        if best_overlap > 1:
            return best_letter
        return ""

    if component == "VESUS":
        if upper in letters:
            return upper
        for letter in letters:
            low_letter = letter.lower()
            if lowered.startswith(f"{low_letter}.") or lowered.startswith(f"{low_letter})"):
                return letter
            if any(token in lowered for token in (f"option {low_letter}", f"choice {low_letter}", f"({low_letter})")):
                return letter
        emotion_keywords = {
            "angry": ("anger", "frustrated", "mad", "furious"),
            "happy": ("joy", "cheerful", "pleased", "delighted"),
            "sad": ("sadness", "melancholy", "depressed", "sorrow"),
            "fearful": ("fear", "anxiety", "scared", "afraid"),
            "monotone": ("flat", "emotionless", "neutral", "bland"),
        }
        for letter, choice in zip(letters, choices):
            choice_text = _choice_body(choice).lower()
            for keywords in emotion_keywords.values():
                if any(keyword in lowered for keyword in keywords) and any(keyword in choice_text for keyword in keywords):
                    return letter
        return ""

    return ""


def audiomarathon_process_results_mcq(doc, results):
    response = results[0] if results else ""
    choices = doc.get("choices") or []
    pred = parse_multi_choice_response(
        response,
        choices,
        component=doc.get("component", ""),
        task_name=doc.get("task_name", ""),
    )
    gold = str(doc["answer"]).strip().upper()
    score = 1.0 if pred == gold else 0.0
    choice_map = {letter: _choice_body(choice) for letter, choice in zip(_choice_letters(len(choices)), choices)}
    item = {
        "score": score,
        "pred": pred,
        "gold": gold,
        "pred_label": choice_map.get(pred, ""),
        "gold_label": choice_map.get(gold, ""),
        "component": doc.get("component", ""),
        "task_name": doc.get("task_name", ""),
        "uniq_id": doc.get("uniq_id", ""),
    }
    return {
        "audiomarathon_f1": dict(item),
        "audiomarathon_acc": dict(item),
        "audiomarathon_score": dict(item),
    }


def _f1_for_label(results, label, pred_key, gold_key):
    true_positive = sum(item[pred_key] == label and item[gold_key] == label for item in results)
    false_positive = sum(item[pred_key] == label and item[gold_key] != label for item in results)
    false_negative = sum(item[pred_key] != label and item[gold_key] == label for item in results)
    denominator = 2 * true_positive + false_positive + false_negative
    return (2 * true_positive / denominator if denominator else 0.0, sum(item[gold_key] == label for item in results))


def audiomarathon_aggregate_f1(results, args=None):
    if not results:
        return 0.0

    task_name = results[0].get("task_name", "")
    component = results[0].get("component", "")
    use_semantic_labels = task_name in {"Speaker gender classification", "Speaker age classification"}
    pred_key = "pred_label" if use_semantic_labels else "pred"
    gold_key = "gold_label" if use_semantic_labels else "gold"

    if component == "HAD":
        true_positive = sum(item.get("pred") == "B" and item.get("gold") == "B" for item in results)
        false_positive = sum(item.get("gold") == "A" and item.get("pred") != "A" for item in results)
        false_negative = sum(item.get("gold") == "B" and item.get("pred") != "B" for item in results)
        denominator = 2 * true_positive + false_positive + false_negative
        score = 2 * true_positive / denominator if denominator else 0.0
        eval_logger.info(f"AudioMarathon official F1 (fake-positive): {score:.5f}")
        return score

    valid = [item for item in results if item.get(pred_key) and item.get(gold_key)]
    if not valid:
        return 0.0

    if component in {"RACE-audio", "GTZAN"}:
        labels = ["A", "B", "C", "D"]
    else:
        labels = sorted({item[gold_key] for item in valid} | {item[pred_key] for item in valid})
    per_label = {label: _f1_for_label(valid, label, pred_key, gold_key) for label in labels}

    if task_name == "Speaker gender classification":
        total_support = sum(support for _, support in per_label.values())
        score = sum(f1 * support for f1, support in per_label.values()) / total_support
        average = "weighted"
    else:
        score = sum(f1 for f1, _ in per_label.values()) / len(per_label)
        average = "macro"

    eval_logger.info(
        f"AudioMarathon official F1 ({average}): {score:.5f}; "
        f"valid predictions {len(valid)}/{len(results)}"
    )
    return score


def audiomarathon_aggregate_acc(results, args=None):
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


def audiomarathon_aggregate_score_mcq(results, args=None):
    """Return the official per-task classification score on a 0-100 scale."""
    return 100.0 * audiomarathon_aggregate_f1(results, args)


def _clean_asr_response(response):
    response = "" if response is None else str(response)
    if not response.strip():
        return ""
    if "assistant\n" in response:
        response = response[response.rfind("assistant\n") + len("assistant\n") :].strip()
    for marker in ("spoken content:", "content:", "transcription:", "transcript:"):
        match = re.search(re.escape(marker), response, flags=re.IGNORECASE)
        if match:
            response = response[match.end() :].strip()
            break
    response = re.sub(r"<transcribed text here>", "", response)
    response = re.sub(r"<sep>.*?($|<|$)", "", response)
    response = re.sub(
        r"^(spoken\s+(?:text|content)|content|transcript|transcription):\s*",
        "",
        response.strip(),
        flags=re.IGNORECASE,
    )
    return response.strip()


def _standardize_asr_text(text):
    text = str(text or "").lower()
    text = re.sub(r'''[.!?,;:"()\[\]{}]''', " ", text)
    text = re.sub(r"[-']", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _word_edit_distance(reference_words, hypothesis_words):
    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, start=1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, start=1):
            current.append(
                min(
                    current[column - 1] + 1,
                    previous[column] + 1,
                    previous[column - 1] + (reference_word != hypothesis_word),
                )
            )
        previous = current
    return previous[-1]


def _official_sample_wer(reference, hypothesis):
    reference_words = _standardize_asr_text(reference).split()
    hypothesis_words = _standardize_asr_text(hypothesis).split()
    if not reference_words or not hypothesis_words:
        return 100.0
    return 100.0 * _word_edit_distance(reference_words, hypothesis_words) / len(reference_words)


def audiomarathon_process_results_asr(doc, results):
    item = {
        "gt": doc["transcript"],
        "pred": _clean_asr_response(results[0] if results else ""),
        "task": "asr_en",
    }
    return {
        "wer": dict(item),
        "word_accuracy": dict(item),
        "audiomarathon_score": dict(item),
    }


def audiomarathon_aggregate_wer(results, args=None):
    valid = [item for item in results if item.get("gt") and item.get("pred")]
    if not valid:
        return 100.0
    eval_logger.info(f"AudioMarathon ASR valid predictions: {len(valid)}/{len(results)}")
    return sum(_official_sample_wer(item["gt"], item["pred"]) for item in valid) / len(valid)


def audiomarathon_aggregate_word_accuracy(results, args=None):
    return 100.0 - audiomarathon_aggregate_wer(results, args)


def audiomarathon_aggregate_score_asr(results, args=None):
    """Return the official ASR score (100 - WER) on the shared 0-100 scale."""
    return audiomarathon_aggregate_word_accuracy(results, args)
