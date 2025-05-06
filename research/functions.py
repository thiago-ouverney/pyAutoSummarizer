import warnings
import pandas as pd
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from scipy.stats import ConstantInputWarning
from pyAutoSummarizer.base.evaluation.base import get_summary_evaluation
from pyAutoSummarizer.base.evaluation.lexical import RougeEvaluator, BLEUEvaluator
from pyAutoSummarizer.base.evaluation.semantic import BERTScoreEvaluator

tqdm.pandas()
warnings.filterwarnings("ignore", category=ConstantInputWarning)


PATH_SUMMEVAL_JSONL = "../data/model_annotations.aligned.jsonl"
LEXICAL_EVAL=[
    RougeEvaluator(),
    BLEUEvaluator(),
    ]
LEXICAL_PREFIX="lexical"
SEMANTIC_PREFIX="semantic"
SEMANTIC_EVAL   =  [BERTScoreEvaluator()]
JOIN_COLS = ["id","model_id"]
# A1 = 0.5
# A2 = 0.5
LEXICAL_COL = f'{LEXICAL_PREFIX}_overall_mean'
SEMANTIC_COL = f'{SEMANTIC_PREFIX}_overall_mean'
NEW_METRIC_COL = 'new_metric_col'
EVAL_COLS = [LEXICAL_PREFIX, SEMANTIC_PREFIX, NEW_METRIC_COL]
# methods = ['pearson']#, 'spearman']
METHODS = ['spearman']
N = 50 #SAMPLE SIZE
HUMAN_COLS = ["exp_"]
FINAL_METRIC = "exp_overall_mean"

def aggregate_dataframe(df,label: str, prefix: str,explode=True) -> pd.DataFrame:
    # Explode a lista na coluna alvo se necessário
    if explode:
        tmp = df[['id', 'model_id', label]].explode(label)
    else:
        tmp = df[['id', 'model_id', label]].copy()
    # Cria colunas separadas para cada métrica
    metrics = pd.json_normalize(tmp[label])
    # Adiciona o prefixo ao nome das colunas
    metrics.columns = [f"{prefix}_{col}" for col in metrics.columns]
    # Volta ao dataframe as colunas 'id' e 'model_id' (identificadoras)
    tmp = pd.concat([tmp[['id', 'model_id']].reset_index(drop=True), metrics], axis=1)
    # Agrupa e calcula média
    grouped = tmp.groupby(['id', 'model_id'], as_index=False).mean()
    # Adiciona a média geral das métricas
    grouped[f'{prefix}_overall_mean'] = grouped.filter(like=f'{prefix}_').mean(axis=1)
    return grouped

def get_metrics_annotations(df: pd.DataFrame) -> pd.DataFrame:
    # Agrega experts e turkers separadamente
    df_exp = aggregate_dataframe(df,'expert_annotations', 'exp')
    df_turk = aggregate_dataframe(df,'turker_annotations', 'turk')

    # Une ambos os resultados em um único DataFrame
    df_metrics = pd.merge(df_exp, df_turk, on=['id', 'model_id'], how='inner')
    return df_metrics


def get_metrics_evaluator(df,evaluators,prefix: str) -> pd.DataFrame:
    tmp = df[['id', 'model_id',"decoded", "references"]].explode("references").copy()
    tmp['eval'] = tmp.progress_apply(
        lambda row: get_summary_evaluation(row['decoded'], row['references'],
                                            evaluators=evaluators
                                        ), axis=1
    )
    df_agg = aggregate_dataframe(tmp, "eval",prefix,explode=False)
    return df_agg


def get_corr_frame(df, eval_cols_preffix , human_cols_preffix, method='pearson'):
    # 1. Seleciona colunas
    eval_columns = [c for c in df.columns if c.startswith(eval_cols_preffix)]
    human_columns = [c for c in df.columns if c.startswith(human_cols_preffix) ]

    # 2. Cria DataFrame de correlações
    correlation_data = {}

    for eval_col in eval_columns:
        row = {}
        for human_col in human_columns:
            x = df[eval_col]
            y = df[human_col]

            if method == 'pearson':
                corr, _ = pearsonr(x, y)
            elif method == 'spearman':
                corr, _ = spearmanr(x, y)
            else:
                raise ValueError("Method must be 'pearson' or 'spearman'")

            row[human_col] = corr
        correlation_data[eval_col] = row

    return pd.DataFrame.from_dict(correlation_data, orient='index')

def get_corr(metrics_frame, eval_cols_preffix , human_cols_preffix, methods=['pearson']):
    correlation_table = pd.DataFrame()
    for method in methods:
        for human_col in human_cols_preffix:
            for eval_col in eval_cols_preffix:
                correlation_table_aux = get_corr_frame(metrics_frame, 
                                                eval_cols_preffix=eval_col, 
                                                human_cols_preffix=human_col, 
                                                method=method)
                correlation_table = pd.concat([correlation_table, correlation_table_aux] , axis=0)  

    return correlation_table 

def get_combinated_metric(avg_summeval_metrics,df_agg,join_cols,A1,A2):
    # Merge the two DataFrames on the specified columns
    combined_df = pd.merge(avg_summeval_metrics, df_agg, on=join_cols, how='inner')
    
    # Calculate the new metric as a weighted sum of the two columns
    combined_df[NEW_METRIC_COL] = A1 * combined_df[LEXICAL_COL] + A2 * combined_df[SEMANTIC_COL]
    
    return combined_df

def get_agg_frame():
    df = pd.read_json(PATH_SUMMEVAL_JSONL, lines=True)
    avg_summeval_metrics = get_metrics_annotations(df)
    df_sample = df.sample(N).copy()
    df_agg_lexical = get_metrics_evaluator(df_sample,LEXICAL_EVAL , LEXICAL_PREFIX)
    df_agg_semantic= get_metrics_evaluator(df_sample,SEMANTIC_EVAL , SEMANTIC_PREFIX )
    df_agg = pd.merge(df_agg_lexical, df_agg_semantic, on= JOIN_COLS)
    return (avg_summeval_metrics,df_agg)

def get_final_corr(avg_summeval_metrics,df_agg,A1,A2):
    metrics_frame = get_combinated_metric(avg_summeval_metrics, df_agg, JOIN_COLS,A1,A2)
    correlation_table = get_corr(metrics_frame, EVAL_COLS, HUMAN_COLS,  METHODS)
    FINAL_CORR = correlation_table.loc[NEW_METRIC_COL,FINAL_METRIC]
    return FINAL_CORR


