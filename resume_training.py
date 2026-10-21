"""Resume after restoring Colab setup, configuration and tokenized data.

Also supports the original Colab bundle, whose ModelTrainer lacks a resume argument.
"""
import inspect
from pathlib import Path
from unittest.mock import patch

from transformers.trainer_utils import get_last_checkpoint

from src.textSummarizer.components import model_trainer as component
from src.textSummarizer.config.configuration import ConfigurationManager


def resume_training():
    config = ConfigurationManager().get_model_trainer_config()
    root = Path(config.root_dir)
    checkpoint = get_last_checkpoint(str(root)) if root.is_dir() else None
    if not checkpoint:
        raise FileNotFoundError(
            f"No checkpoint found in {root}. If the session ended before step 250, "
            "the original notebook has no saved training state to resume."
        )
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "config.json")
    missing = [name for name in required if not (Path(checkpoint) / name).is_file()]
    if missing:
        raise RuntimeError(
            f"Checkpoint may be incomplete: {checkpoint}; missing {missing}. "
            "Inspect the preceding checkpoint before resuming."
        )
    print(f"Resuming training from {checkpoint}", flush=True)
    job = component.ModelTrainer(config)
    if "resume_from_checkpoint" in inspect.signature(job.train).parameters:
        return job.train(resume_from_checkpoint=checkpoint)

    # Compatibility for the source ZIP already uploaded to an active Colab session.
    original_train = component.Trainer.train

    def train_from_checkpoint(trainer, *args, **kwargs):
        kwargs["resume_from_checkpoint"] = checkpoint
        return original_train(trainer, *args, **kwargs)

    with patch.object(component.Trainer, "train", train_from_checkpoint):
        return job.train()


if __name__ == "__main__":
    resume_training()
