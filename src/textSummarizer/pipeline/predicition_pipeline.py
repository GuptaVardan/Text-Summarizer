"""Load one local checkpoint and reuse it for successive requests."""
from pathlib import Path
from threading import Lock

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.textSummarizer.config.configuration import ConfigurationManager

GENERATION_KWARGS = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
MAX_INPUT_TOKENS = 1024


class PredictionPipeline:
    def __init__(self, model_path=None, tokenizer_path=None):
        if model_path is None or tokenizer_path is None:
            config = ConfigurationManager().get_model_evaluation_config()
            model_path = model_path or config.model_path
            tokenizer_path = tokenizer_path or config.tokenizer_path
        for path in (model_path, tokenizer_path):
            if not Path(path).is_dir():
                raise FileNotFoundError(
                    f"Missing checkpoint directory: {path}. Run python main.py first."
                )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, local_files_only=True)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_path, local_files_only=True
        ).to(self.device)
        self.model.eval()
        self._lock = Lock()

    def predict(self, text):
        # Bound concurrent GPU memory use in this process.
        with self._lock, torch.inference_mode():
            inputs = self.tokenizer(
                text, max_length=MAX_INPUT_TOKENS, truncation=True, return_tensors="pt"
            ).to(self.device)
            output = self.model.generate(**inputs, **GENERATION_KWARGS)
            return self.tokenizer.decode(output[0], skip_special_tokens=True)
