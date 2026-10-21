# Changes

## Training and evaluation corrections

- Corrected `Trainer.train_dataset` from the SAMSum test split to the training split. Validation remains separate, and final evaluation uses all 819 test examples.
- Removed a decoded-text replacement that inserted spaces between every character before ROUGE scoring. Prediction strings now reach the scorer unchanged.
- Filter missing, non-string, or blank dialogue/summary pairs before tokenization. The local SAMSum CSV has one empty training dialogue; 14,731 valid training pairs remain. Save cleaning counts with the dataset artifacts.
- Use an explicit shared ROUGE protocol: arithmetic mean per-example F1, stemming enabled, on a 0-100 scale. The pipeline and standalone benchmark share the implementation. Earlier ten-example or corrupted evaluations are not comparable to this protocol.
- Added benchmark output with predictions, references, dataset fingerprint, software/hardware context, and warm single-input latency. Timing excludes model loading and HTTP overhead.
- Recorded the completed Colab T4 training and full-test reports under `results/colab-t4/`. These reports document a corrected run, not a controlled before/after comparison or reproduction of earlier reported numbers.

## Training reliability

- Updated target-tokenization and Trainer processing APIs.
- Retained gradient accumulation of 16 with physical batch size 1; enabled gradient checkpointing and disabled model cache during training, restoring it for saved inference artifacts.
- Save intermediate checkpoints every 250 optimizer updates and retain up to two. Added checkpoint resume support and a Drive-backed Colab workflow.
- Removed the incompatible `ensure` import that prevented Python 3.13 imports.

## Inference service

- Reuse the local model/tokenizer across requests instead of loading them for every prediction.
- Disable gradients and use model evaluation mode during generation.
- Accept validated JSON input and return a structured summary response with appropriate failure codes.
- Remove the HTTP training trigger; run training through the CLI.
- Add regression tests, CPU CI, deployment instructions, and artifact exclusions.

Historical notebooks remain unchanged as experiment records. No custom attention mechanism, measured GPU memory reduction, HTTP response-time result, or production load-test result is claimed.
