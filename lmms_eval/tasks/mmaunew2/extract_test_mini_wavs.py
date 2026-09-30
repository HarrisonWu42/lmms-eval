"""Extract MMAU-v05.15.25 test-mini WAV files from its Parquet file."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_PARQUET_PATH = Path(
    "/Users/andylubowen/Desktop/MMAU-v05.15.25/test-mini/test_mini.parquet"
)
DEFAULT_JSON_PATH = Path(
    "/Users/andylubowen/Desktop/EIT-code/AudioMCQevaluator/mmau-test-mini.json"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/andylubowen/Desktop/EIT-code/test-mini-audios"
)


def _load_expected_audio_files(json_path: Path) -> dict[str, str]:
    with json_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON array in {json_path}")

    expected: dict[str, str] = {}
    for record in records:
        if not isinstance(record, dict):
            raise ValueError(f"Invalid non-object record in {json_path}")

        sample_id = str(record.get("id", "")).strip()
        audio_id = str(record.get("audio_id", "")).strip()
        if not sample_id or not audio_id:
            raise ValueError(f"Record is missing id or audio_id: {record!r}")
        if sample_id in expected:
            raise ValueError(f"Duplicate MMAU sample id: {sample_id}")

        filename = Path(audio_id).name
        if not filename:
            raise ValueError(f"Invalid audio_id for {sample_id}: {audio_id!r}")
        expected[sample_id] = filename
    return expected


def _parse_attributes(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Invalid MMAU other_attributes value: {value!r}")


def _audio_bytes(value: Any, sample_id: str) -> bytes:
    if not isinstance(value, dict):
        raise ValueError(f"Invalid context value for {sample_id}: {type(value).__name__}")
    content = value.get("bytes")
    if content is None:
        raise ValueError(f"Parquet context has no embedded bytes for {sample_id}")
    content = bytes(content)
    if not content:
        raise ValueError(f"Parquet context has empty audio bytes for {sample_id}")
    return content


def _is_wav_header(content: bytes) -> bool:
    return len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WAVE"


def _is_valid_wav(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    with path.open("rb") as handle:
        return _is_wav_header(handle.read(12))


def _write_wav(content: bytes, output_path: Path, sample_id: str) -> None:
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    source_path = output_path.with_name(f".{output_path.name}.source")
    temporary_path.unlink(missing_ok=True)
    source_path.unlink(missing_ok=True)

    try:
        if _is_wav_header(content):
            temporary_path.write_bytes(content)
        else:
            ffmpeg = shutil.which("ffmpeg")
            if ffmpeg is None:
                raise RuntimeError(
                    f"Sample {sample_id} contains compressed audio; ffmpeg is required "
                    "to create a real WAV file"
                )

            source_path.write_bytes(content)
            completed = subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-y",
                    "-i",
                    str(source_path),
                    "-map",
                    "0:a:0",
                    "-c:a",
                    "pcm_s16le",
                    "-f",
                    "wav",
                    str(temporary_path),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                detail = completed.stderr.strip() or "unknown ffmpeg error"
                raise RuntimeError(f"Failed to convert {sample_id} to WAV: {detail}")

        if not _is_valid_wav(temporary_path):
            raise RuntimeError(f"Extraction produced an invalid WAV for {sample_id}")
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
        source_path.unlink(missing_ok=True)


def extract_wavs(
    parquet_path: Path,
    json_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> tuple[int, int]:
    expected = _load_expected_audio_files(json_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet = pq.ParquetFile(parquet_path)
    required_columns = {"context", "other_attributes"}
    missing = sorted(required_columns - set(parquet.schema_arrow.names))
    if missing:
        raise ValueError(f"Parquet is missing required columns: {missing}")

    extracted = 0
    skipped = 0
    seen: set[str] = set()
    for batch in parquet.iter_batches(
        batch_size=32,
        columns=["context", "other_attributes"],
    ):
        for row in batch.to_pylist():
            attributes = _parse_attributes(row["other_attributes"])
            sample_id = str(attributes.get("id", "")).strip()
            if sample_id not in expected:
                raise ValueError(f"Unexpected Parquet sample id: {sample_id!r}")
            if sample_id in seen:
                raise ValueError(f"Duplicate Parquet sample id: {sample_id}")
            seen.add(sample_id)

            output_path = output_dir / expected[sample_id]
            if _is_valid_wav(output_path) and not overwrite:
                skipped += 1
                continue

            content = _audio_bytes(row["context"], sample_id)
            _write_wav(content, output_path, sample_id)
            extracted += 1

    missing_ids = sorted(set(expected) - seen)
    if missing_ids:
        raise ValueError(
            f"Parquet is missing {len(missing_ids)} JSON samples; first: {missing_ids[0]}"
        )

    invalid_outputs = [
        filename
        for filename in expected.values()
        if not _is_valid_wav(output_dir / filename)
    ]
    if invalid_outputs:
        raise RuntimeError(
            f"Extraction left {len(invalid_outputs)} missing or empty WAV files; "
            f"first: {invalid_outputs[0]}"
        )
    return extracted, skipped


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract MMAU-v05.15.25 test-mini audio bytes to WAV files."
    )
    parser.add_argument("--parquet", type=Path, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing non-empty WAV files.",
    )
    args = parser.parse_args()

    extracted, skipped = extract_wavs(
        parquet_path=args.parquet.expanduser(),
        json_path=args.json.expanduser(),
        output_dir=args.output_dir.expanduser(),
        overwrite=args.overwrite,
    )
    print(
        f"MMAU test-mini WAV extraction complete: "
        f"{extracted} extracted, {skipped} existing files kept."
    )


if __name__ == "__main__":
    main()
