from pyAutoSummarizer.base.evaluation.lexical import RougeEvaluator, BLEUEvaluator
from pyAutoSummarizer.base.evaluation.semantic import BERTScoreEvaluator

PATH_SUMMEVAL_JSONL = "../data/model_annotations.aligned.jsonl"
LEXICAL_EVAL=[
    RougeEvaluator(),
    BLEUEvaluator(),
    ]
LEXICAL_PREFIX="lexical"
SEMANTIC_PREFIX="semantic"
SEMANTIC_EVAL   =  [BERTScoreEvaluator()]
JOIN_COLS = ["id","model_id"]
LEXICAL_COL = f'{LEXICAL_PREFIX}_overall_mean'
SEMANTIC_COL = f'{SEMANTIC_PREFIX}_overall_mean'
HIBRITY_QUALITY_SCORE = 'hybrid_quality_score'
EVAL_COLS = [LEXICAL_PREFIX, SEMANTIC_PREFIX, HIBRITY_QUALITY_SCORE]
METHODS = ['spearman']
N = 2 #SAMPLE SIZE
HUMAN_COLS = ["exp_"]
FINAL_METRIC = "exp_overall_mean"
# FINAL_METRIC = HIBRITY_QUALITY_SCORE
