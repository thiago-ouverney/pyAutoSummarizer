from .base import EvaluationMetric
from typing import Dict
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from moverscore import word_mover_score  # type: ignore

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
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import nltk; nltk.download("punkt")

class SentenceBertEvaluator(EvaluationMetric):
    def __init__(self, model_name: str = "all-mpnet-base-v2", agg: str = "mean"):
        super().__init__("SentenceBERT")
        self.model = SentenceTransformer(model_name)
        self.agg = agg

    def _aggregate(self, embeds: np.ndarray) -> np.ndarray:
        if self.agg == "mean":
            return embeds.mean(axis=0)
        elif self.agg == "max":
            return embeds.max(axis=0)
        elif self.agg == "median":
            return np.median(embeds, axis=0)
        else:
            raise ValueError("agg must be mean|max|median")

    def evaluate(self, reference: str, generated: str):
        ref_sents = nltk.sent_tokenize(reference)
        gen_sents = nltk.sent_tokenize(generated)

        ref_embeds = self.model.encode(ref_sents, convert_to_numpy=True, show_progress_bar=False)
        gen_embeds = self.model.encode(gen_sents, convert_to_numpy=True, show_progress_bar=False)

        ref_vec = self._aggregate(ref_embeds)
        gen_vec = self._aggregate(gen_embeds)

        cosine = float(cosine_similarity([ref_vec], [gen_vec])[0, 0])
        # opcional – matriz frase-a-frase
        sim_matrix = util.cos_sim(
            self.model.encode(ref_sents, convert_to_tensor=True),
            self.model.encode(gen_sents, convert_to_tensor=True)
        ).cpu().numpy()

        return {
            "sentence_bert_cosine": float(cosine),
            "sentence_bert_sim_avg": float(sim_matrix.mean()),
            "sentence_bert_sim_max": float(sim_matrix.max())
        }

# class MoverScoreEvaluator(EvaluationMetric):
#     """F‑score do MoverScore v2."""

#     def __init__(self, model_type: str = "bert-base-uncased"):
#         super().__init__("MoverScore")
#         if word_mover_score is None:
#             raise ImportError("moverscore não encontrado. `pip install moverscore`. ")
#         self.model_type = model_type

#     # pylint: disable=arguments-differ
#     def evaluate(self, reference: str, generated: str) -> Dict[str, float]:
#         # MoverScore retorna uma lista de escores (uma entrada → uma saída)
#         score_list = word_mover_score(
#             [reference], [generated], model_type=self.model_type, n_gram=1, batch_size=1, remove_subwords=True
#         )
#         return {"moverscore": float(score_list[0])}
