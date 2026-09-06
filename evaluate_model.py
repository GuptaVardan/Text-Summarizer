from pathlib import Path
import time

import evaluate
import torch
from datasets import load_from_disk
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


MODEL_PATH = Path("artifacts/model_trainer/pegasus-samsum-model")
TOKENIZER_PATH = Path("artifacts/model_trainer/tokenizer")
DATA_PATH = Path("artifacts/data_transformation/samsum_dataset")
SAMPLE_SIZE = 100
BATCH_SIZE = 4


def evaluate_model():
    missing = [path for path in (MODEL_PATH, TOKENIZER_PATH, DATA_PATH) if not path.exists()]
    if missing:
        missing_paths = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(
            f"Evaluation artifacts are missing: {missing_paths}. "
            "Run the training pipeline before benchmarking."
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_PATH).to(device)
    model.eval()

    dataset = load_from_disk(DATA_PATH)["test"]
    sample = dataset.select(range(min(SAMPLE_SIZE, len(dataset))))
    metric = evaluate.load("rouge")
    total_seconds = 0.0
    prediction_count = 0

    with torch.inference_mode():
        for start in range(0, len(sample), BATCH_SIZE):
            dialogues = sample[start : start + BATCH_SIZE]["dialogue"]
            inputs = tokenizer(
                dialogues,
                max_length=1024,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(device)

            if device.type == "cuda":
                torch.cuda.synchronize()
            started = time.perf_counter()
            generated = model.generate(
                **inputs,
                num_beams=8,
                length_penalty=0.8,
                max_length=128,
            )
            if device.type == "cuda":
                torch.cuda.synchronize()
            total_seconds += time.perf_counter() - started

            predictions = tokenizer.batch_decode(generated, skip_special_tokens=True)
            metric.add_batch(
                predictions=predictions,
                references=sample[start : start + BATCH_SIZE]["summary"],
            )
            prediction_count += len(predictions)

    scores = metric.compute()
    print(f"device: {device}")
    print(f"samples: {prediction_count}")
    print(f"ROUGE-1: {scores['rouge1']:.6f}")
    print(f"ROUGE-2: {scores['rouge2']:.6f}")
    print(f"ROUGE-L: {scores['rougeL']:.6f}")
    print(f"average inference latency: {total_seconds / prediction_count * 1000:.3f} ms/summary")


if __name__ == "__main__":
    evaluate_model()
