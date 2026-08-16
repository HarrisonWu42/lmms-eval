import argparse
import json
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = TASK_DIR / "manifests"


def find_default_root():
    for parent in (Path.cwd(), *Path.cwd().parents, TASK_DIR, *TASK_DIR.parents):
        candidate = parent / "datasets" / "AudioMarathon"
        if candidate.exists():
            return candidate.resolve()
    return (Path.cwd() / "datasets" / "AudioMarathon").resolve()


def read_json(path):
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def exists(root, rel_path):
    return (root / rel_path).exists()


def first_existing(root, *rel_paths):
    for rel_path in rel_paths:
        rel_path = Path(rel_path)
        if exists(root, rel_path):
            return rel_path
    return None


def choice_fields(item):
    choices = {}
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        key = f"choice_{letter.lower()}"
        if key in item and item[key] not in (None, ""):
            choices[letter] = str(item[key]).strip()
    return choices


def normalize_gold(item, choices):
    raw = item.get("answer", item.get("answer_gt", ""))
    raw = str(raw).strip()
    upper = raw.upper()
    if upper in choices:
        return upper
    for letter, value in choices.items():
        if raw == value or raw.lower() == value.lower():
            return letter
    raise ValueError(f"Cannot map answer {raw!r} to choices {choices!r}")


def make_mcq_record(component, task_name, audio_path, question, choices, answer, uniq_id, extra=None):
    record = {
        "uniq_id": str(uniq_id),
        "component": component,
        "task_name": task_name,
        "audio_path": str(audio_path),
        "question": str(question),
        "choices": [f"{letter}. {choices[letter]}" for letter in choices],
        "answer": answer,
        "answer_text": choices.get(answer, ""),
    }
    if extra:
        record.update(extra)
    return record


def manifest_alimeeting(root):
    source = root / "AliMeeting" / "meetingqa" / "Test_Ali_longform.json"
    records = []
    for item in read_json(source):
        audio_rel = Path("AliMeeting") / item["audio_paths"][0]
        if not exists(root, audio_rel):
            continue
        records.append(
            {
                "uniq_id": item["meeting_id"],
                "meeting_id": item["meeting_id"],
                "component": "AliMeeting",
                "task_name": "Meeting summarization",
                "audio_path": str(audio_rel),
                "prompt": item["prompt"],
                "reference_summary": item["reference_summary"],
                "judge": item.get("judge", {}),
            }
        )
    return records


def manifest_librispeech(root):
    base = root / "librispeech-long"
    records = []
    for audio_path in sorted(base.glob("*/*/*/*.flac")):
        transcript_path = audio_path.with_suffix(".txt")
        if not transcript_path.exists():
            continue
        split = audio_path.relative_to(base).parts[0]
        records.append(
            {
                "uniq_id": f"{split}_{audio_path.stem}",
                "component": "LibriSpeech-long",
                "task_name": "Automatic speech recognition",
                "audio_path": str(audio_path.relative_to(root)),
                "transcript": transcript_path.read_text(encoding="utf-8").strip(),
                "split": split,
            }
        )
    return records


def manifest_race(root):
    records = []
    for item in read_json(root / "race_audio" / "race_benchmark.json"):
        audio_rel = Path("race_audio") / item["audio_path"]
        if not exists(root, audio_rel):
            continue
        choices = {letter: value for letter, value in zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", item["options"])}
        answer = normalize_gold(item, choices)
        records.append(
            make_mcq_record(
                "RACE-audio",
                "Reading comprehension from audio",
                audio_rel,
                item["question"],
                choices,
                answer,
                f"{item['article_id']}_{item['question_idx']}",
                {"article_id": item.get("article_id"), "question_idx": item.get("question_idx")},
            )
        )
    return records


def manifest_had(root):
    data = read_json(root / "HAD" / "concatenated_audio" / "had_audio_classification_task.json")
    records = []
    for item in data["samples"]:
        audio_rel = Path("HAD") / "concatenated_audio" / item["path"]
        if not exists(root, audio_rel):
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        records.append(make_mcq_record("HAD", "Half-truth audio detection", audio_rel, item["question"], choices, answer, item["uniq_id"]))
    return records


def manifest_gtzan(root):
    records = []
    source = root / "GTZAN" / "concatenated_audio" / "music_genre_classification_meta.json"
    for item in read_json(source):
        audio_rel = first_existing(
            root,
            Path("GTZAN") / "concatenated_audio" / item["path"],
            Path("GTZAN") / "concatenated_audio" / "wav" / item["path"],
        )
        if audio_rel is None:
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        records.append(make_mcq_record("GTZAN", "Music genre classification", audio_rel, item["question"], choices, answer, item["uniq_id"]))
    return records


def manifest_tau(root):
    records = []
    source = root / "TAU" / "concatenated_resampled" / "acoustic_scene_task_meta.json"
    for item in read_json(source):
        audio_rel = Path("TAU") / "concatenated_resampled" / item["path"]
        if not exists(root, audio_rel):
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        records.append(make_mcq_record("TAU", "Acoustic scene classification", audio_rel, item["question"], choices, answer, item["uniq_id"]))
    return records


def manifest_vesus(root):
    source = root / "VESUS" / "audio_emotion_dataset_filtered.json"
    data = read_json(source)
    items = data["data"] if isinstance(data, dict) and "data" in data else data
    records = []
    for item in items:
        audio_rel = Path("VESUS") / item["path"]
        if not exists(root, audio_rel):
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        records.append(make_mcq_record("VESUS", "Emotion recognition", audio_rel, item["question"], choices, answer, item["uniq_id"]))
    return records


def manifest_slue(root):
    records = []
    for item in read_json(root / "SLUE" / "merged_audio_data.json"):
        audio_rel = Path("SLUE") / item["path"]
        if not exists(root, audio_rel):
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        records.append(make_mcq_record("SLUE", "Speech named entity reasoning", audio_rel, item["question"], choices, answer, item["uniq_id"]))
    return records


def manifest_desed(root):
    base = root / "DESED" / "DESED_dataset" / "concatenated_audio"
    labels = sorted(path.name for path in base.iterdir() if path.is_dir() and not path.name.startswith("."))
    choices = {chr(ord("A") + index): label.replace("_", " ") for index, label in enumerate(labels)}
    records = []
    for label in labels:
        answer = chr(ord("A") + labels.index(label))
        for audio_path in sorted((base / label).glob("*.wav")):
            audio_rel = audio_path.relative_to(root)
            question = "What sound event is represented in this audio segment?"
            records.append(make_mcq_record("DESED", "Sound event detection", audio_rel, question, choices, answer, audio_path.stem, {"event_label": label}))
    return records


def manifest_voxceleb_gender(root):
    records = []
    source = root / "VoxCeleb" / "concatenated_audio" / "gender_id_task_meta.json"
    for item in read_json(source):
        audio_rel = first_existing(
            root,
            Path("VoxCeleb") / "concatenated_audio" / item["path"],
            Path("VoxCeleb") / "concatenated_audio" / "wav" / item["path"],
        )
        if audio_rel is None:
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        records.append(make_mcq_record("VoxCeleb", "Speaker gender classification", audio_rel, item["question"], choices, answer, item["uniq_id"]))
    return records


def manifest_voxceleb_age(root):
    records = []
    source = root / "VoxCeleb" / "concatenated_audio" / "age_classification_task_meta.json"
    for item in read_json(source):
        audio_rel = first_existing(
            root,
            Path("VoxCeleb") / "concatenated_audio" / item["path"],
            Path("VoxCeleb") / "concatenated_audio" / "wav" / item["path"],
        )
        if audio_rel is None:
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        records.append(make_mcq_record("VoxCeleb", "Speaker age classification", audio_rel, item["question"], choices, answer, item["uniq_id"]))
    return records


BUILDERS = {
    "alimeeting_summary": manifest_alimeeting,
    "librispeech_long": manifest_librispeech,
    "race_audio": manifest_race,
    "had": manifest_had,
    "gtzan": manifest_gtzan,
    "tau": manifest_tau,
    "vesus": manifest_vesus,
    "slue": manifest_slue,
    "desed": manifest_desed,
    "voxceleb_gender": manifest_voxceleb_gender,
    "voxceleb_age": manifest_voxceleb_age,
}


def main():
    parser = argparse.ArgumentParser(description="Build AudioMarathon JSONL manifests for lmms-eval.")
    parser.add_argument("--root", type=Path, default=find_default_root(), help="Path to datasets/AudioMarathon.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output manifest directory.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"AudioMarathon root does not exist: {root}")

    for name, builder in BUILDERS.items():
        records = builder(root)
        output = args.out_dir / f"{name}.jsonl"
        write_jsonl(output, records)
        print(f"{name}: wrote {len(records)} records to {output}")


if __name__ == "__main__":
    main()
