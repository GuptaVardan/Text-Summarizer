"""Score a local checkpoint and benchmark warm, serial, single-dialogue inference."""
import argparse
import hashlib
import json
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

import torch
from datasets import load_from_disk
from src.textSummarizer.utils.rouge_metrics import rouge_scores

from src.textSummarizer.pipeline.predicition_pipeline import (
    GENERATION_KWARGS, MAX_INPUT_TOKENS, PredictionPipeline,
)


def percentile(values, fraction):
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def evaluate_model(args):
    paths = (args.model_path, args.tokenizer_path, args.data_path)
    missing = [str(path) for path in paths if not path.is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing evaluation artifacts: {', '.join(missing)}. "
            "Run python main.py to train a checkpoint before benchmarking."
        )
    dataset = load_from_disk(str(args.data_path))['test']
    if not len(dataset):
        raise ValueError('The test split is empty.')
    if args.sample_size:
        dataset = dataset.shuffle(seed=args.seed).select(range(min(args.sample_size, len(dataset))))
    predictor = PredictionPipeline(args.model_path, args.tokenizer_path)
    for _ in range(args.warmup):
        predictor.predict(dataset[0]['dialogue'])

    records, timings = [], []
    for row in dataset:
        if predictor.device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        summary = predictor.predict(row['dialogue'])
        if predictor.device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        timings.append(elapsed)
        records.append({'id': row.get('id'), 'prediction': summary,
                        'reference': row['summary'], 'latency_seconds': elapsed})
        if len(records) % 25 == 0:
            print(f"Evaluated {len(records)}/{len(dataset)} examples", flush=True)

    config_path = args.model_path / 'config.json'
    report = {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'model_path': str(args.model_path.resolve()),
        'model_config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
        'tokenizer_path': str(args.tokenizer_path.resolve()),
        'dataset_path': str(args.data_path.resolve()),
        'dataset_fingerprint': dataset._fingerprint,
        'split': 'test', 'samples': len(records), 'sampling_seed': args.seed,
        'device': str(predictor.device),
        'hardware': torch.cuda.get_device_name() if predictor.device.type == 'cuda' else platform.processor(),
        'platform': platform.platform(),
        'versions': {name: version(name) for name in ('torch', 'transformers', 'datasets', 'rouge-score')},
        'generation': {**GENERATION_KWARGS, 'max_input_tokens': MAX_INPUT_TOKENS},
        'rouge_f1_percent': rouge_scores([r['prediction'] for r in records], [r['reference'] for r in records]),
        'rouge_method': 'Arithmetic mean of per-example F1, stemming enabled; 0-100 scale.',
        'latency_seconds': {'mean': statistics.mean(timings), 'p50': percentile(timings, .5),
                            'p95': percentile(timings, .95), 'warmup_requests': args.warmup,
                            'batch_size': 1,
                            'scope': 'Warm serial pipeline: tokenization, generation, decoding; excludes model load, HTTP, network and queueing.'},
        'training_provenance': 'Confirm checkpoint was trained only on train; this report cannot establish that.',
    }
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    with (args.output / 'predictions.jsonl').open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')
    print(json.dumps(report, indent=2))
    return report


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-path', type=Path, default=Path('artifacts/model_trainer/pegasus-samsum-model'))
    parser.add_argument('--tokenizer-path', type=Path, default=Path('artifacts/model_trainer/tokenizer'))
    parser.add_argument('--data-path', type=Path, default=Path('artifacts/data_transformation/samsum_dataset'))
    parser.add_argument('--sample-size', type=int, default=0, help='0 evaluates the entire test split.')
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', type=Path, default=Path('artifacts/benchmark'))
    args = parser.parse_args()
    if args.sample_size < 0 or args.warmup < 0:
        parser.error('--sample-size and --warmup must be nonnegative')
    return args


if __name__ == '__main__':
    evaluate_model(parse_args())
