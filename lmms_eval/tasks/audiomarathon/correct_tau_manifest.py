"""Create a corrected AudioMarathon TAU manifest without changing the original.

The released long-audio construction placed both ``metro`` and
``metro_station`` recordings under the ``metro`` label.  The original scene is
still recoverable from filenames such as ``metro_station_paris-81_segment_2``.
This script repairs only those rows and validates the resulting manifest.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


TASK_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = TASK_DIR / "manifests" / "tau.jsonl"
DEFAULT_OUTPUT = TASK_DIR / "manifests" / "tau_corrected.jsonl"
METRO_STATION_TEXT = "Metro station - metro_station"
TAU_LABELS = (
    "street_pedestrian",
    "street_traffic",
    "metro_station",
    "public_square",
    "shopping_mall",
    "airport",
    "metro",
    "tram",
    "park",
    "bus",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def choice_body(choice: str) -> str:
    prefix, separator, body = choice.partition(". ")
    return body if separator and len(prefix.strip()) == 1 else choice.strip()


def choice_label(choice: str) -> str:
    body = choice_body(choice)
    return body.rsplit(" - ", 1)[-1].strip().lower()


def answer_label(record: dict) -> str:
    return str(record["answer_text"]).rsplit(" - ", 1)[-1].strip().lower()


def filename_label(record: dict) -> str:
    filename = Path(record["audio_path"]).name.lower()
    for label in TAU_LABELS:
        if filename.startswith(f"{label}_"):
            return label
    raise ValueError(f"Could not infer a TAU label from filename {filename!r}")


def correct_record(record: dict) -> tuple[dict, str]:
    """Correct one metro-station row and report how its choices changed."""
    choices = list(record["choices"])
    station_indexes = [
        index for index, choice in enumerate(choices) if choice_label(choice) == "metro_station"
    ]

    if station_indexes:
        station_index = station_indexes[0]
        correction_type = "selected_existing_metro_station_choice"
    else:
        gold_letter = str(record["answer"]).strip().upper()
        gold_index = ord(gold_letter) - ord("A")
        if not 0 <= gold_index < len(choices):
            raise ValueError(f"Invalid gold answer {gold_letter!r} for {record['audio_path']}")
        if choice_label(choices[gold_index]) != "metro":
            raise ValueError(
                f"Expected the old gold choice to be metro for {record['audio_path']}, "
                f"got {choices[gold_index]!r}"
            )
        letter = chr(ord("A") + gold_index)
        choices[gold_index] = f"{letter}. {METRO_STATION_TEXT}"
        station_index = gold_index
        correction_type = "replaced_old_metro_gold_choice"

    corrected = dict(record)
    corrected["choices"] = choices
    corrected["answer"] = chr(ord("A") + station_index)
    corrected["answer_text"] = METRO_STATION_TEXT
    return corrected, correction_type


def validate(records: list[dict]) -> Counter:
    labels = Counter()
    mismatches = []

    for line_number, record in enumerate(records, start=1):
        choices = record.get("choices") or []
        answer = str(record.get("answer", "")).strip().upper()
        answer_index = ord(answer) - ord("A") if len(answer) == 1 else -1
        if not 0 <= answer_index < len(choices):
            raise ValueError(f"Line {line_number}: answer {answer!r} does not select a choice")

        selected_text = choice_body(str(choices[answer_index]))
        if selected_text != record.get("answer_text"):
            raise ValueError(
                f"Line {line_number}: answer_text {record.get('answer_text')!r} "
                f"does not match selected choice {selected_text!r}"
            )
        if len(set(choices)) != len(choices):
            raise ValueError(f"Line {line_number}: duplicate choices in {record['audio_path']}")

        gold_label = answer_label(record)
        source_label = filename_label(record)
        labels[gold_label] += 1
        if gold_label != source_label:
            mismatches.append((line_number, source_label, gold_label, record["audio_path"]))

    if mismatches:
        preview = "; ".join(map(str, mismatches[:5]))
        raise ValueError(f"Found {len(mismatches)} filename/answer mismatches: {preview}")
    if len(labels) != 10:
        raise ValueError(f"Expected 10 ground-truth classes, found {len(labels)}: {dict(labels)}")
    return labels


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    if input_path == output_path:
        raise ValueError("Input and output must be different so the original manifest is preserved")

    source_lines = input_path.read_text(encoding="utf-8").splitlines(keepends=True)
    output_lines = []
    output_records = []
    correction_counts = Counter()
    unchanged_count = 0

    for line_number, source_line in enumerate(source_lines, start=1):
        if not source_line.strip():
            output_lines.append(source_line)
            continue
        record = json.loads(source_line)
        if filename_label(record) == "metro_station":
            corrected, correction_type = correct_record(record)
            correction_counts[correction_type] += 1
            output_lines.append(json.dumps(corrected, ensure_ascii=False) + "\n")
            output_records.append(corrected)
        else:
            output_lines.append(source_line)
            output_records.append(record)
            unchanged_count += 1

    labels = validate(output_records)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(output_lines), encoding="utf-8")

    print(f"input={input_path}")
    print(f"output={output_path}")
    print(f"rows={len(output_records)}")
    print(f"corrected={sum(correction_counts.values())}")
    print(f"unchanged={unchanged_count}")
    for name, count in sorted(correction_counts.items()):
        print(f"{name}={count}")
    print("label_counts=" + json.dumps(dict(sorted(labels.items())), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
