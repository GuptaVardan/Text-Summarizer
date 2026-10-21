# Conversational Text Summarization

An end-to-end Python/PyTorch pipeline for SAMSum dialogue summarization: data ingestion, cleaning, Pegasus fine-tuning, held-out evaluation, and FastAPI inference.

## Recorded results

A completed Google Colab run trained `google/pegasus-cnn_dailymail` for one epoch on **14,731 valid training examples**, with 818 validation examples. Evaluation covered all **819 held-out test examples**.

| Measurement | Result |
| --- | ---: |
| ROUGE-1 F1 | 45.26 |
| ROUGE-2 F1 | 21.61 |
| ROUGE-L F1 | 35.60 |
| Mean warm inference latency | 0.792 s |
| Median warm inference latency | 0.721 s |
| p95 warm inference latency | 1.175 s |
| Hardware | NVIDIA Tesla T4 |
| Training duration | 63 min 50 s |

ROUGE uses mean per-example F1 with stemming on a 0-100 scale. Latency measures serial, batch-size-one tokenization, generation, and decoding after three warmups. **It excludes model loading, HTTP overhead, network, and queueing; it is not an API response-time benchmark.** ROUGE measures reference overlap, not factual accuracy.

The [benchmark report](results/colab-t4/metrics.json) and [training report](results/colab-t4/train_results.json) were captured from the completed Colab run and supplied by the project owner. See [result provenance](results/colab-t4/README.md) for versions, limitations, and reproduction details. Checkpoint weights and per-example predictions are not included in Git.

## Pipeline

```text
SAMSum archive -> clean dialogue/summary pairs -> tokenize inputs and labels
             -> Pegasus training on train, validation during training
             -> generation and ROUGE on test -> local checkpoint -> FastAPI
```

The original SAMSum splits contain 14,732 training, 818 validation, and 819 test examples. One training row has an empty dialogue; filtering invalid pairs leaves 14,731 training examples. Cleaning counts are saved with preprocessing artifacts. No test examples are used for training.

Generation uses standard tokenizer-generated padding masks, eight-beam search, length penalty 0.8, and maximum sequence length 128. Inputs are truncated to 1,024 tokens. Training uses physical batch size 1, gradient accumulation 16, and gradient checkpointing. These settings manage training memory; no before/after memory benchmark is claimed.

## Setup

Use Python 3.11 or newer:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell: .\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-dev.txt
```

`requirements-dev.txt` includes the runtime dependencies and the HTTP test client. Install just `requirements.txt` if tests are not needed. On Windows, use `.\.venv\Scripts\python.exe` explicitly if `python` resolves to another installation.

## Train and resume

```bash
python main.py
```

Configure paths in `config/config.yaml` and hyperparameters in `params.yaml`. The CLI runs ingestion, transformation, training, then full-test ROUGE evaluation. The supplied archive contains the dataset splits; the pipeline removes invalid text pairs before tokenization.

Full Pegasus fine-tuning should run on a GPU rather than the development laptop's CPU. For Google Colab, open [pegasus_finetuning_colab.ipynb](pegasus_finetuning_colab.ipynb), select a GPU, and run its cells. It clones this repository, saves model artifacts to Drive, and resumes the latest checkpoint when present. **No dataset or source ZIP upload is required by the current notebook.**

Checkpoints are saved every 250 optimizer updates, with at most two intermediate checkpoints retained. After a runtime reset, restore the same configuration and cleaned/tokenized data, then run:

```bash
python resume_training.py
```

This restores saved training state. Work after the latest checkpoint is lost; an interrupted first 250 steps have no scheduled checkpoint. Keep the same data, batch size, accumulation, and package versions when resuming. The notebook includes Drive recovery steps. Resume argument forwarding is tested; a full interrupted GPU training run has not been independently reproduced locally.

Final artifacts:

```text
artifacts/model_trainer/pegasus-samsum-model/
artifacts/model_trainer/tokenizer/
artifacts/model_trainer/train_results.json
```

The Colab notebook changes the model root to `MyDrive/textsummarizer-verification/model_trainer`. Download the final model and tokenizer folders to the paths above for local inference. Weights are not bundled with the repository.

## Serve

```bash
python -m uvicorn app:app --host 127.0.0.1 --port 8080
```

Open `http://127.0.0.1:8080/docs`, or POST JSON to `/predict`:

```bash
curl -X POST http://127.0.0.1:8080/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"Alice: Can we meet at 6? Bob: Yes, at the cafe."}'
```

Response: `{"summary":"..."}`. Requests require nonblank text of at most 20,000 characters. The model/tokenizer load on first use and are cached per process; generation is serialized per instance. Missing checkpoints return HTTP 503, malformed requests return 422, and generation failures return 500. Training runs separately from the API.

CPU inference is supported, with no laptop latency guarantee. This service has not been load-tested or benchmarked over HTTP. The input limit and model truncation can omit context, and summaries may contain factual errors.

Optional Docker inference:

```bash
docker build -t text-summarizer .
docker run --rm -p 8080:8080 -v "$PWD/artifacts:/app/artifacts" text-summarizer
```

Supply the local model/tokenizer directories through the artifacts mount. The Docker image has not been built in the current development environment.

## Evaluate and benchmark

```bash
python evaluate_model.py --sample-size 10 --output artifacts/benchmark-smoke
python evaluate_model.py --output artifacts/benchmark
```

The first command samples ten test examples with seed 42; the second evaluates the full test split. Progress prints every 25 examples. Both require a trained checkpoint and tokenized dataset. Outputs include `metrics.json` and `predictions.jsonl`, recording example IDs, predictions, references, timings, versions, generation settings, and dataset fingerprint.

The training-stage ROUGE CSV and standalone benchmark use the same stemming, F1 aggregation, and 0-100 scale. The pipeline CSV uses batched generation, while the latency benchmark uses single-input generation; generated outputs may differ numerically across environments and batching. Evaluate the base model with the same protocol before claiming improvement from fine-tuning. A model configuration hash is not a weight-file checksum or proof of training provenance.

## Tests and maintenance

```bash
python -m unittest discover -s tests -v
```

Tests cover split selection, resume forwarding, text cleaning, unchanged decoded predictions, metric scaling, cached inference, API validation/error responses, and real forward/backward and beam-search execution using a tiny random Pegasus. They do not establish full-model quality or throughput. GitHub Actions runs the CPU tests.

See [CHANGELOG.md](CHANGELOG.md) for the corrected split/evaluation bugs and service changes. The notebooks under `research/` retain historical outputs and older APIs; use the Python pipeline or current Colab notebook for reproduction. Dataset licensing is separate from the repository's code license; see the SAMSum dataset card before reuse.
