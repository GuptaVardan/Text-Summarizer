"""Shared ROUGE protocol: mean per-example F1, stemming, 0-100 scale."""
import statistics

from rouge_score import rouge_scorer


def rouge_scores(predictions, references):
    if not predictions or len(predictions) != len(references):
        raise ValueError("Predictions and references must have the same nonzero length.")
    names = ("rouge1", "rouge2", "rougeL")
    scorer = rouge_scorer.RougeScorer(names, use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]
    return {
        name: 100 * statistics.mean(row[name].fmeasure for row in scores)
        for name in names
    }


class RougeMetric:
    """Batch accumulator for the training pipeline's evaluation stage."""

    def __init__(self):
        self.predictions = []
        self.references = []

    def add_batch(self, *, predictions, references):
        if len(predictions) != len(references):
            raise ValueError("Predictions and references must have equal lengths.")
        self.predictions.extend(predictions)
        self.references.extend(references)

    def compute(self):
        return rouge_scores(self.predictions, self.references)
