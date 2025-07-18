from pyAutoSummarizer.base.evaluation.lexical import RougeEvaluator, BLEUEvaluator, METEOREvaluator
from pyAutoSummarizer.base.evaluation.semantic import BERTScoreEvaluator, SentenceBertEvaluator
from pyAutoSummarizer.base.evaluation.factual import FactCCEvaluator

PATH_SUMMEVAL_JSONL = "../data/model_annotations.aligned.jsonl"
LEXICAL_EVAL=[
    RougeEvaluator(),
    BLEUEvaluator(),
    METEOREvaluator()
    ]
LEXICAL_PREFIX="lexical"
SEMANTIC_PREFIX="semantic"
SEMANTIC_EVAL   =  [
    BERTScoreEvaluator(),
    SentenceBertEvaluator(),
    ]
FACTUAL_PREFIX="factual"
FACTUAL_EVAL   =  [
    FactCCEvaluator(),
    ]

JOIN_COLS = ["id","model_id"]
LEXICAL_COL = f'{LEXICAL_PREFIX}_overall_mean'
SEMANTIC_COL = f'{SEMANTIC_PREFIX}_overall_mean'
HIBRITY_QUALITY_SCORE = 'hybrid_quality_score'
EVAL_COLS = [LEXICAL_PREFIX, SEMANTIC_PREFIX, FACTUAL_PREFIX, HIBRITY_QUALITY_SCORE]
METHODS = ['spearman']
N = 2 #SAMPLE SIZE
HUMAN_COLS = ["exp_"]
FINAL_METRIC = "exp_overall_mean"
FILE_PATH_DF_AGG = "../data/df_agg.csv"
FILE_PATH_AVG_SUMMEVAL_METRICS = "../data/avg_summeval_metrics.csv"
# FINAL_METRIC = HIBRITY_QUALITY_SCORE
