"""Load the released MMAU mini JSON, including its mixed category types."""

import json

import datasets


class MMAUOfficial(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")

    def _info(self):
        fields = {
            name: datasets.Value("string")
            for name in (
                "id",
                "audio_id",
                "question",
                "answer",
                "dataset",
                "task",
                "split",
                "category",
                "sub-category",
                "difficulty",
            )
        }
        fields["choices"] = datasets.Sequence(datasets.Value("string"))
        return datasets.DatasetInfo(features=datasets.Features(fields))

    def _split_generators(self, dl_manager):
        files = dl_manager.download_and_extract(self.config.data_files)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"paths": files["test"]},
            )
        ]

    def _generate_examples(self, paths):
        for path in paths:
            with open(path, encoding="utf-8") as handle:
                records = json.load(handle)
            for record in records:
                # Arrow needs one column type. This metadata is not scored;
                # leave question, choices, answer, and grouping fields intact.
                category = record["category"]
                if isinstance(category, list):
                    (record["category"],) = category
                yield record["id"], record
