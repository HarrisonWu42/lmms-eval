import argparse
import glob
import json
import random
from pathlib import Path

TASK_DIR = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = TASK_DIR / "manifests"

OFFICIAL_DATASET_REVISION = "ef00ea0458a57cb75b32d18ddb3b3619c4388fec"

# Records evaluated by the official Qwen2.5-Omni baseline at the pinned Hub
# revision.  The upstream scripts balance HAD and VoxCeleb gender with
# random.seed(42), filter two speakers' happy VESUS examples, and skip missing
# audio.  Consequently the evaluation protocol uses 5,774 records even though
# the benchmark README describes 6,563 runnable records before these filters.
EXPECTED_RECORD_COUNTS = {
    "librispeech_long": 204,
    "race_audio": 820,
    "had": 630,
    "gtzan": 120,
    "tau": 1145,
    "vesus": 178,
    "slue": 490,
    "desed": 254,
    "voxceleb_gender": 976,
    "voxceleb_age": 957,
}


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


def manifest_librispeech(root):
    base = root / "librispeech-long" / "test-clean"
    records = []
    for audio_path in sorted(base.glob("*/*/*.flac")):
        transcript_path = audio_path.with_suffix(".txt")
        if not transcript_path.exists():
            continue
        split = base.name
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
    records = []
    base = root / "HAD" / "concatenated_audio"
    choices = {"A": "real", "B": "fake"}
    question = (
        "Listen to this audio clip carefully. Is this audio completely authentic (real) "
        "or does it contain any artificially synthesized segments (fake)? If it is "
        "completely real, answer 'a'. If it contains any fake segments, answer 'b'. "
        "Answer with only 'a' or 'b'."
    )
    for label, answer in (("real", "A"), ("fake", "B")):
        # glob.glob intentionally matches the official loader's directory order.
        for filename in glob.glob(str(base / label / "*.wav")):
            audio_path = Path(filename)
            audio_rel = audio_path.relative_to(root)
            records.append(
                make_mcq_record(
                    "HAD",
                    "Half-truth audio detection",
                    audio_rel,
                    question,
                    choices,
                    answer,
                    audio_rel.with_suffix("").as_posix(),
                )
            )
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
    source = root / "VESUS" / "audio_emotion_dataset.json"
    items = read_json(source)
    records = []
    for item in items:
        audio_rel = Path("VESUS") / item["path"]
        if not exists(root, audio_rel):
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        records.append(
            make_mcq_record(
                "VESUS",
                "Emotion recognition",
                audio_rel,
                item["question"],
                choices,
                answer,
                item["uniq_id"],
                {"person_id": item.get("person_id"), "emotion_label": item.get("emotion_label", "")},
            )
        )
    return records


def manifest_slue(root):
    records = []
    for item in read_json(root / "SLUE" / "merged_audio_data.json"):
        audio_rel = Path("SLUE") / item["path"]
        if not exists(root, audio_rel):
            continue
        choices = choice_fields(item)
        answer = normalize_gold(item, choices)
        uniq_id = Path(item["path"]).with_suffix("").as_posix()
        records.append(make_mcq_record("SLUE", "Speech named entity reasoning", audio_rel, item["question"], choices, answer, uniq_id))
    return records


def manifest_desed(root):
    base = root / "DESED" / "DESED_dataset" / "concatenated_audio"
    source = base / "desed_sound_event_detection_task.json"
    records = []
    for item in read_json(source)["tasks"]:
        audio_rel = Path("DESED") / "DESED_dataset" / "concatenated_audio" / item["path"]
        if not exists(root, audio_rel):
            continue
        choices = {str(letter): str(value) for letter, value in item["choices"].items()}
        answer = normalize_gold(item, choices)
        records.append(
            make_mcq_record(
                "DESED",
                "Sound event detection",
                audio_rel,
                item["question"],
                choices,
                answer,
                item.get("uniq_id", Path(item["path"]).stem),
                {"event_label": item.get("correct_event", item.get("primary_event", ""))},
            )
        )
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


def balanced_sample(records, labels, seed):
    grouped = {label: [] for label in labels}
    for record in records:
        label = record["answer_text"].strip().lower()
        if label in grouped:
            grouped[label].append(record)

    target = min(len(grouped[label]) for label in labels)
    rng = random.Random(seed)
    selected = []
    for label in labels:
        group = grouped[label]
        selected.extend(rng.sample(group, target) if len(group) > target else group)
    rng.shuffle(selected)
    return selected


def select_official_qwen_records(name, records, seed):
    """Apply the sample selection in src/Qwen_2.5_Omni/Others."""
    if name == "had":
        return balanced_sample(records, ("real", "fake"), seed)
    if name == "voxceleb_gender":
        return balanced_sample(records, ("male", "female"), seed)
    if name == "vesus":
        return [
            record
            for record in records
            if not (
                str(record.get("person_id", "")).strip() in {"2", "10"}
                and str(record.get("emotion_label", "")).strip().lower() == "happy"
            )
        ]
    if name in {"tau", "voxceleb_age"}:
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        return shuffled
    return records


def main():
    parser = argparse.ArgumentParser(
        description=f"Build official AudioMarathon JSONL manifests for lmms-eval ({OFFICIAL_DATASET_REVISION})."
    )
    parser.add_argument("--root", type=Path, default=find_default_root(), help="Path to datasets/AudioMarathon.")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR, help="Output manifest directory.")
    parser.add_argument("--seed", type=int, default=42, help="Official Qwen baseline sampling seed.")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"AudioMarathon root does not exist: {root}")

    stale_manifest = args.out_dir / "alimeeting_summary.jsonl"
    if stale_manifest.exists():
        stale_manifest.unlink()

    total_records = 0
    for name, builder in BUILDERS.items():
        records = select_official_qwen_records(name, builder(root), args.seed)
        expected = EXPECTED_RECORD_COUNTS[name]
        if len(records) != expected:
            raise RuntimeError(
                f"{name}: expected {expected} runnable records from official revision "
                f"{OFFICIAL_DATASET_REVISION}, found {len(records)}"
            )
        output = args.out_dir / f"{name}.jsonl"
        write_jsonl(output, records)
        print(f"{name}: wrote {len(records)} records to {output}")
        total_records += len(records)
    print(f"official Qwen evaluation total: {total_records}")


if __name__ == "__main__":
    main()
