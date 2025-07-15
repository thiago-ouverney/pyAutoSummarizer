# pyAutoSummarizer/evaluation/lexical.py
from __future__ import annotations

from typing import Dict, List
from .base import EvaluationMetric

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk, string, re

from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score

# ──────────────────────────────────────────────────────────────
# Configura recursos NLTK apenas na primeira importação
# ──────────────────────────────────────────────────────────────
for pkg in ("punkt", "stopwords"):
    try:
        nltk.data.find(f"tokenizers/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOP_SET = set(stopwords.words("english"))
_PUNCT_REGEX = re.compile(f"[{re.escape(string.punctuation)}]")

def _preprocess(
    text: str,
    *,
    remove_stopwords: bool = False,
    remove_punct: bool = True,
) -> List[str]:
    """Tokeniza, coloca em minúsculas, remove pontuação e, opcionalmente, stop-words."""
    text = text.lower()
    if remove_punct:
        text = _PUNCT_REGEX.sub(" ", text)
    tokens = word_tokenize(text)
    if remove_stopwords:
        tokens = [tok for tok in tokens if tok not in _STOP_SET]
    return tokens


# ──────────────────────────────────────────────────────────────
# AVALIADORES LEXICAIS
# ──────────────────────────────────────────────────────────────
class RougeEvaluator(EvaluationMetric):
    def __init__(self, *, remove_stopwords: bool = True, remove_punct: bool = True):
        super().__init__("ROUGE")
        self.remove_stopwords = remove_stopwords
        self.remove_punct = remove_punct
        # Tokenizer interno do rouge_scorer é suficiente após pré-processamento
        self.scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=True)

    def evaluate(self, reference: str, generated: str) -> Dict[str, float]:
        ref = " ".join(_preprocess(reference,
                                   remove_stopwords=self.remove_stopwords,
                                   remove_punct=self.remove_punct))
        gen = " ".join(_preprocess(generated,
                                   remove_stopwords=self.remove_stopwords,
                                   remove_punct=self.remove_punct))
        scores = self.scorer.score(ref, gen)
        return {
            "rouge1_f1": scores["rouge1"].fmeasure,
            "rougeL_f1": scores["rougeL"].fmeasure,
        }


class BLEUEvaluator(EvaluationMetric):
    def __init__(self, *, remove_stopwords: bool = True, remove_punct: bool = True):
        super().__init__("BLEU")
        self.remove_stopwords = remove_stopwords
        self.remove_punct = remove_punct
        self._smooth = SmoothingFunction().method4

    def evaluate(self, reference: str, generated: str) -> Dict[str, float]:
        ref_tokens = _preprocess(reference,
                                 remove_stopwords=self.remove_stopwords,
                                 remove_punct=self.remove_punct)
        gen_tokens = _preprocess(generated,
                                 remove_stopwords=self.remove_stopwords,
                                 remove_punct=self.remove_punct)
        bleu = sentence_bleu([ref_tokens], gen_tokens, smoothing_function=self._smooth)
        return {"bleu": bleu}


class METEOREvaluator(EvaluationMetric):
    def __init__(self, *, remove_stopwords: bool = True, remove_punct: bool = True):
        super().__init__("METEOR")
        self.remove_stopwords = remove_stopwords
        self.remove_punct = remove_punct

    def evaluate(self, reference: str, generated: str) -> Dict[str, float]:
        ref_tokens = _preprocess(
            reference,
            remove_stopwords=self.remove_stopwords,
            remove_punct=self.remove_punct,
        )
        gen_tokens = _preprocess(
            generated,
            remove_stopwords=self.remove_stopwords,
            remove_punct=self.remove_punct,
        )

        # >>> NÃO use " ".join(...); passe as listas diretamente <<<
        meteor = meteor_score([ref_tokens], gen_tokens)
        return {"meteor": meteor}
