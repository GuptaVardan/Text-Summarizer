# Colab T4 run

These JSON reports were supplied by the project owner from a completed Google Colab GPU run. They are preserved as reported, rather than regenerated or claimed as locally reproduced measurements.

- `metrics.json`: full 819-example test evaluation, eight beams, mean per-example ROUGE F1 with stemming, 0-100 scale.
- `train_results.json`: one epoch, 14,731 training examples, 818 validation examples, gradient accumulation 16, 3,830.493 seconds of reported training runtime.
- `cleaning_counts.json`: documented split counts reconstructed from the local CSV inspection and training report; not an original uploaded Colab artifact.

The benchmark includes three warmups and serial single-dialogue inference on a Tesla T4. It times tokenization, generation, and decoding. Model loading, HTTP, network and queueing are excluded. The config hash identifies configuration content, not checkpoint weights. No confidence interval, base-model comparison, HTTP benchmark, or memory-saving benchmark is provided.

The recorded environment was PyTorch 2.11.0+cu130, Transformers 5.20.1, Datasets 4.8.5, and rouge-score 0.1.2. Training/checkpoint files and `predictions.jsonl` remain outside Git and must be retained by the owner to support reproduction. A published model download is not currently provided, so cloning this repository alone does not reproduce the reported scores without retraining or obtaining those artifacts.

Reproduction:

1. Use the corrected training split and preprocessing in the repository. The uncleaned training split has 14,732 rows; one empty dialogue is removed. Preserve the dataset partitions and training configuration.
2. Train the configured Pegasus checkpoint for one epoch with batch size 1 and accumulation 16. The current Colab notebook saves artifacts to Drive.
3. Run `python evaluate_model.py --output artifacts/benchmark` on all test examples. Use the exact recorded package versions and GPU context for a closer comparison; exact timings and model outputs can vary.
4. Retain the generated predictions, full checkpoint, training logs, and reports. The JSON report alone cannot prove split provenance or prediction correctness.

The reported timestamp is UTC. These results were shared on 22 October 2026 in the owner's Asia/Kolkata timezone.
