import os
import json
from pathlib import Path
from src.textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk

from src.textSummarizer.entity import DataTransformationConfig


class DataTransformation:
    def __init__(self,config:DataTransformationConfig):
        self.config=config
        self.tokenizer=AutoTokenizer.from_pretrained(config.tokenizer_name)

    @staticmethod
    def valid_example(example):
        return all(
            isinstance(example.get(field), str) and example[field].strip()
            for field in ("dialogue", "summary")
        )

    def convert_examples_to_features(self,example_batch):
        input_encodings = self.tokenizer(example_batch['dialogue'] , max_length = 1024, truncation = True )

        target_encodings = self.tokenizer(
            text_target=example_batch['summary'], max_length=128, truncation=True
        )

        return {
            'input_ids' : input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }

    def convert(self):
        dataset_samsum = load_from_disk(self.config.data_path)
        counts = {}
        for split in dataset_samsum:
            before = len(dataset_samsum[split])
            dataset_samsum[split] = dataset_samsum[split].filter(self.valid_example)
            counts[split] = {"before": before, "after": len(dataset_samsum[split]),
                             "removed": before - len(dataset_samsum[split])}
            if not len(dataset_samsum[split]):
                raise ValueError(f"No valid dialogue-summary pairs in {split}.")
        logger.info("Data cleaning counts: %s", counts)
        Path(self.config.root_dir).mkdir(parents=True, exist_ok=True)
        (Path(self.config.root_dir) / "cleaning_counts.json").write_text(
            json.dumps(counts, indent=2), encoding="utf-8"
        )
        dataset_samsum_pt = dataset_samsum.map(self.convert_examples_to_features, batched = True)
        dataset_samsum_pt.save_to_disk(os.path.join(self.config.root_dir,"samsum_dataset"))



