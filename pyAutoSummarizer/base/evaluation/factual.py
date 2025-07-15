from .base import EvaluationMetric
from transformers import BertTokenizer, BertForSequenceClassification
import torch
from typing import Optional
MODEL_NAME = "manueldeprada/FactCC"
class FactCCEvaluator(EvaluationMetric):
    """
    Avalia factualidade com o modelo FactCC.
    
    Retorna um score contínuo ∈[0, 1] (ou ∈[-1, 1] se `signed=True`)
    que pode ser usado diretamente na otimização Bayesiana
    e na correlação com julgamentos humanos.
    """
    def __init__(
        self,
        name: str = "FactCC",
        device: Optional[str] = None,
        signed: bool = False,           # se True, devolve score em [-1,1]
    ):
        super().__init__(name)
        self.tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
        self.model = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        self.model.eval()
        self.label2id = {v: k for k, v in self.model.config.id2label.items()}
        self.id2label = self.model.config.id2label
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.signed = signed

    @torch.inference_mode()
    def _predict(self, source: str, summary: str) -> tuple[str, float]:
        """Devolve (label_str, probabilidade_do_label)."""
        encoded = self.tokenizer(
            source,
            summary,
            truncation="only_first",
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(self.device)

        logits = self.model(**encoded).logits[0]
        probs = torch.softmax(logits, dim=-1)
        label_id = torch.argmax(probs).item()
        label_str = self.id2label[label_id]
        return label_str, probs[label_id].item()

    def evaluate(self, reference: str, generated: str) -> dict:
        label, p_label = self._predict(reference, generated)

        # probabilidade de o RESUMO ser factual (label CORRECT)
        p_correct = (
            p_label if label == "CORRECT" else 1.0 - p_label
        )  # ∈ [0,1]

        if self.signed:
            # mapeia para [-1,1] (útil se quiser penalizar muito incorreções)
            score = 2 * p_correct - 1
        else:
            score = p_correct

        return {
            "factcc_score": score,     # escalar contínuo usado em otimização
            # "factcc_label": label,     # rótulo para depuração / análise
        }
