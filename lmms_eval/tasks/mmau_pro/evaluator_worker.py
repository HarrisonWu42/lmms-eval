"""Short-lived worker for one MMAU-Pro model-backed evaluator."""

import argparse
import json
import os
from pathlib import Path

import utils as mmau_pro_utils


def _write_json(path, value):
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluator", choices=("qwen_judge", "nvembed"))
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    config = payload.get("config")
    items = payload.get("items")
    if not isinstance(config, dict) or not isinstance(items, list):
        raise ValueError("MMAU-Pro evaluator worker received an invalid payload")
    mmau_pro_utils._EVALUATOR_CONFIG.update(config)

    if args.evaluator == "qwen_judge":
        records = mmau_pro_utils._judge_open_items(items)
    else:
        records = mmau_pro_utils._match_closed_items(items)

    if len(records) != len(items):
        raise RuntimeError(
            f"MMAU-Pro {args.evaluator} produced {len(records)} records "
            f"for {len(items)} samples"
        )
    _write_json(args.output, records)


if __name__ == "__main__":
    main()
