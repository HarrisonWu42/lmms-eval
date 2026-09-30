"""Dataset loader for the AudioMCQ MMAU test-mini JSON manifest."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import datasets


class MMAUNew2TestMini(datasets.GeneratorBasedBuilder):
    """Load the manifest while normalizing its mixed ``category`` field."""

    VERSION = datasets.Version("1.0.0")

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "id": datasets.Value("string"),
                    "audio_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "choices": datasets.Sequence(datasets.Value("string")),
                    "answer": datasets.Value("string"),
                    "dataset": datasets.Value("string"),
                    "task": datasets.Value("string"),
                    "split": datasets.Value("string"),
                    "category": datasets.Value("string"),
                    "sub-category": datasets.Value("string"),
                    "difficulty": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        if not self.config.data_files or "test" not in self.config.data_files:
            raise ValueError("mmaunew2 test-mini requires data_files.test")

        downloaded = dl_manager.download_and_extract(self.config.data_files)
        test_files = downloaded["test"]
        if isinstance(test_files, (str, Path)):
            test_files = [test_files]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"paths": list(test_files)},
            )
        ]

    @staticmethod
    def _normalize_category(value: Any, sample_id: str) -> str:
        if isinstance(value, str):
            return value
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            if len(value) == 1 and isinstance(value[0], str):
                return value[0]
        raise ValueError(
            f"MMAU test-mini sample {sample_id!r} has invalid category: {value!r}"
        )

    def _generate_examples(self, paths):
        seen_ids: set[str] = set()
        for path in paths:
            with open(path, "r", encoding="utf-8") as handle:
                records = json.load(handle)
            if not isinstance(records, list):
                raise ValueError(f"MMAU test-mini manifest must be a JSON array: {path}")

            for index, record in enumerate(records):
                if not isinstance(record, Mapping):
                    raise ValueError(f"MMAU test-mini row {index} is not an object")

                normalized = dict(record)
                sample_id = str(normalized.get("id", "")).strip()
                if not sample_id:
                    raise ValueError(f"MMAU test-mini row {index} has no id")
                if sample_id in seen_ids:
                    raise ValueError(f"Duplicate MMAU test-mini id: {sample_id}")
                seen_ids.add(sample_id)
                normalized["category"] = self._normalize_category(
                    normalized.get("category"), sample_id
                )
                yield sample_id, normalized
