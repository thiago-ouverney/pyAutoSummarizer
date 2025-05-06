from .base import EvaluationMetric
from questeval.questeval_metric import QuestEval

class QuestEvalEvaluator(EvaluationMetric):
    def __init__(self):
        super().__init__('QuestEval')
        self.questeval = QuestEval()

    def evaluate(self, reference: str, generated: str) -> dict:
        scores = self.questeval.compute_all(pred=generated, answers=reference)
        return {
            "questeval_fscore": scores["fscore"]
        }
