from .base import EvaluationMetric
import bert_score
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

class BERTScoreEvaluator(EvaluationMetric):
    def __init__(self, lang="en"):
        super().__init__('BERTScore')
        self.lang = lang

    def evaluate(self, reference: str, generated: str) -> dict:
        P, R, F1 = bert_score.score([generated], [reference], lang=self.lang, rescale_with_baseline=True)
        return {
            "bert_score_precision": P[0].item(),
            "bert_score_recall": R[0].item(),
            "bert_score_f1": F1[0].item()
        }
