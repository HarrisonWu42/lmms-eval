#!/usr/bin/env python3
"""Extract MMAU-v05.15.25 test-mini WAVs from its Parquet file.

The official JSON is the manifest and determines the expected sample IDs and
output filenames. The Parquet file is used only by this one-time materializer;
``mmaunew2`` subsequently reads the generated WAV files directly.

Default usage::

    python3 materialize_test_mini_wavs.py

Choose the Parquet directory and WAV output directory::

    python3 materialize_test_mini_wavs.py \
        --input-dir /path/to/MMAU-v05.15.25/test-mini \
        --output-dir /path/to/test-mini-audios

Install the only external dependency if needed::

    python3 -m pip install pyarrow

Existing non-empty WAV files are skipped, so interrupted extraction can be
resumed. Pass ``--overwrite`` to rewrite them.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

try:
    import pyarrow.parquet as pq
except ModuleNotFoundError:
    pq = None


DEFAULT_JSON_PATH = Path(
    "/Users/andylubowen/Desktop/EIT-code/AudioMCQevaluator/mmau-test-mini.json"
)
DEFAULT_PARQUET_PATH = Path(
    "/Users/andylubowen/Desktop/MMAU-v05.15.25/test-mini/test_mini.parquet"
)
DEFAULT_OUTPUT_DIR = Path(
    "/Users/andylubowen/Desktop/EIT-code/AudioMCQevaluator/test-mini-audios"
)
EXPECTED_ROW_COUNT = 1_000


def _load_manifest(json_path: Path) -> dict[str, str]:
    with json_path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)

    if not isinstance(records, list):
        raise ValueError(f"{json_path} must contain a JSON array")
    if len(records) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Unexpected JSON row count: {len(records)} != {EXPECTED_ROW_COUNT}"
        )

    manifest: dict[str, str] = {}
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"{json_path}: row {index} is not an object")

        sample_id = str(record.get("id", "")).strip()
        filename = Path(str(record.get("audio_id", "")).strip()).name
        if not sample_id or not filename:
            raise ValueError(
                f"{json_path}: row {index} requires non-empty id and audio_id"
            )
        if sample_id in manifest:
            raise ValueError(f"Duplicate JSON id {sample_id!r}")
        manifest[sample_id] = filename
    return manifest


def _metadata_id(value: Any, parquet_path: Path) -> str:
    if isinstance(value, Mapping):
        metadata = value
    else:
        try:
            metadata = json.loads(str(value))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid other_attributes JSON in {parquet_path}: {value!r}"
            ) from exc

    if not isinstance(metadata, Mapping):
        raise ValueError(f"Invalid other_attributes object in {parquet_path}")
    return str(metadata.get("id", "")).strip()


def _write_bytes_atomic(audio_bytes: Any, output_path: Path) -> None:
    if audio_bytes is None:
        raise ValueError("embedded audio bytes are missing")
    payload = bytes(audio_bytes)
    if not payload:
        raise ValueError("embedded audio bytes are empty")

    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_bytes(payload)
    os.replace(temporary_path, output_path)


def materialize(
    json_path: Path,
    parquet_path: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> None:
    if pq is None:
        raise SystemExit(
            "Missing dependency 'pyarrow'. Run: python3 -m pip install pyarrow"
        )
    if not json_path.is_file():
        raise FileNotFoundError(f"JSON manifest not found: {json_path}")
    if not parquet_path.is_file():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    manifest = _load_manifest(json_path)
    parquet_file = pq.ParquetFile(parquet_path)
    missing_columns = {"context", "other_attributes"} - set(
        parquet_file.schema_arrow.names
    )
    if missing_columns:
        raise ValueError(
            f"{parquet_path} is missing columns: {sorted(missing_columns)}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    seen_ids: set[str] = set()
    written = 0
    skipped = 0

    print(
        f"[MMAU test-mini] {len(manifest)} JSON rows, "
        f"{parquet_file.metadata.num_rows} Parquet rows -> {output_dir}",
        flush=True,
    )
    for batch in parquet_file.iter_batches(
        batch_size=16,
        columns=["context", "other_attributes"],
        use_threads=False,
    ):
        for row in batch.to_pylist():
            sample_id = _metadata_id(row.get("other_attributes"), parquet_path)
            if sample_id not in manifest:
                continue
            if sample_id in seen_ids:
                raise ValueError(f"Duplicate Parquet id {sample_id!r}")
            seen_ids.add(sample_id)

            audio = row.get("context")
            if not isinstance(audio, Mapping):
                raise ValueError(
                    f"Parquet sample {sample_id!r} has no context audio struct"
                )

            filename = manifest[sample_id]
            parquet_audio_path = audio.get("path")
            parquet_filename = (
                Path(str(parquet_audio_path)).name if parquet_audio_path else ""
            )
            if parquet_filename and parquet_filename != filename:
                raise ValueError(
                    f"Audio filename mismatch for {sample_id!r}: "
                    f"JSON={filename!r}, Parquet={parquet_filename!r}"
                )

            output_path = output_dir / filename
            if (
                not overwrite
                and output_path.is_file()
                and output_path.stat().st_size > 0
            ):
                skipped += 1
                continue

            try:
                _write_bytes_atomic(audio.get("bytes"), output_path)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to write Parquet sample {sample_id!r}"
                ) from exc
            written += 1
            if written % 100 == 0:
                print(
                    f"[MMAU test-mini] wrote {written}, skipped {skipped}",
                    flush=True,
                )

    missing_ids = sorted(set(manifest) - seen_ids)
    if missing_ids:
        preview = ", ".join(missing_ids[:5])
        raise ValueError(
            f"{len(missing_ids)} JSON ids were not found in Parquet; "
            f"first ids: {preview}"
        )

    missing_wavs = [
        filename
        for filename in manifest.values()
        if not (output_dir / filename).is_file()
        or (output_dir / filename).stat().st_size == 0
    ]
    if missing_wavs:
        preview = ", ".join(missing_wavs[:5])
        raise RuntimeError(
            f"{len(missing_wavs)} WAV files are still missing; first files: {preview}"
        )

    print(
        f"[MMAU test-mini] complete: {written} written, {skipped} skipped, "
        f"{len(manifest)} ready",
        flush=True,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract MMAU-v05.15.25 test-mini WAVs from Parquet."
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"JSON manifest (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing test_mini.parquet. Cannot be used together "
            "with --parquet-path."
        ),
    )
    parser.add_argument(
        "--parquet-path",
        type=Path,
        default=None,
        help=f"Source Parquet (default: {DEFAULT_PARQUET_PATH})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"WAV output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite existing non-empty WAV files.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.input_dir is not None and args.parquet_path is not None:
        raise SystemExit("Use either --input-dir or --parquet-path, not both.")

    parquet_path = (
        args.input_dir.expanduser() / "test_mini.parquet"
        if args.input_dir is not None
        else (
            args.parquet_path.expanduser()
            if args.parquet_path is not None
            else DEFAULT_PARQUET_PATH
        )
    )
    materialize(
        json_path=args.json_path.expanduser(),
        parquet_path=parquet_path,
        output_dir=args.output_dir.expanduser(),
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
