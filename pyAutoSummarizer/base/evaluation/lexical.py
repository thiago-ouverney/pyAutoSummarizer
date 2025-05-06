
## pyAutoSummarizer/evaluation/lexical.py
from .base import EvaluationMetric
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

class RougeEvaluator(EvaluationMetric):
    def __init__(self):
        super().__init__('ROUGE')
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)

    def evaluate(self, reference: str, generated: str) -> dict:
        scores = self.scorer.score(reference, generated)
        return {
            "rouge1_f1": scores['rouge1'].fmeasure,
            "rougeL_f1": scores['rougeL'].fmeasure
        }

class BLEUEvaluator(EvaluationMetric):
    def __init__(self):
        super().__init__('BLEU')

    def evaluate(self, reference: str, generated: str) -> dict:
        smoothie = SmoothingFunction().method4
        bleu_score = sentence_bleu([reference.split()], generated.split(), smoothing_function=smoothie)
        return {"bleu": bleu_score}

class METEOREvaluator(EvaluationMetric):
    def __init__(self):
        super().__init__('METEOR')

    def evaluate(self, reference: str, generated: str) -> dict:
        score = meteor_score([reference], generated)
        return {"meteor": score}
