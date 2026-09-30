"""Load the MMAU test-mini JSON after normalizing its mixed category field."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import datasets


class MMAULMMSTestMini(datasets.GeneratorBasedBuilder):
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
            raise ValueError("mmaulmms test-mini requires data_files.test")

        files = dl_manager.download_and_extract(self.config.data_files)["test"]
        if isinstance(files, (str, Path)):
            files = [files]
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"paths": list(files)},
            )
        ]

    def _generate_examples(self, paths):
        seen_ids = set()
        for path in paths:
            with open(path, "r", encoding="utf-8") as source_file:
                records = json.load(source_file)
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

                category = normalized.get("category")
                if isinstance(category, Sequence) and not isinstance(category, (str, bytes)):
                    if len(category) != 1 or not isinstance(category[0], str):
                        raise ValueError(
                            f"MMAU test-mini sample {sample_id!r} has invalid category: {category!r}"
                        )
                    category = category[0]
                if not isinstance(category, str):
                    raise ValueError(
                        f"MMAU test-mini sample {sample_id!r} has invalid category: {category!r}"
                    )
                normalized["category"] = category

                yield sample_id, normalized
