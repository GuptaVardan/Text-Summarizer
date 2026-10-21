"""Evaluate generated summaries on the complete held-out test split."""
import csv
from pathlib import Path

import torch
from datasets import load_from_disk
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.textSummarizer.entity import ModelEvaluationConfig
from src.textSummarizer.pipeline.predicition_pipeline import GENERATION_KWARGS, MAX_INPUT_TOKENS
from src.textSummarizer.utils.rouge_metrics import RougeMetric


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    @staticmethod
    def generate_batch_sized_chunks(elements, batch_size):
        for start in range(0, len(elements), batch_size):
            yield elements[start:start + batch_size]

    def calculate_metric_on_test_ds(
        self, dataset, metric, model, tokenizer, batch_size=2, device=None,
        column_text="dialogue", column_summary="summary",
    ):
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        texts = dataset[column_text]
        references = dataset[column_summary]
        model.eval()
        for start in tqdm(range(0, len(texts), batch_size), desc="Evaluating summaries"):
            inputs = tokenizer(
                texts[start:start + batch_size], max_length=MAX_INPUT_TOKENS,
                truncation=True, padding=True, return_tensors="pt",
            )
            with torch.inference_mode():
                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device),
                    **GENERATION_KWARGS,
                )
            decoded = [tokenizer.decode(summary, skip_special_tokens=True)
                       for summary in summaries]
            metric.add_batch(
                predictions=decoded, references=references[start:start + batch_size]
            )
        return metric.compute()

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path, local_files_only=True)
        model = AutoModelForSeq2SeqLM.from_pretrained(
            self.config.model_path, local_files_only=True
        ).to(device)
        dataset = load_from_disk(self.config.data_path)["test"]
        scores = self.calculate_metric_on_test_ds(dataset, RougeMetric(), model, tokenizer, device=device)
        output = Path(self.config.metric_file_name)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(scores))
            writer.writeheader()
            writer.writerow(scores)
        return scores
