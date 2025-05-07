import warnings
import pandas as pd
from tqdm.auto import tqdm
from scipy.stats import spearmanr, pearsonr
from scipy.stats import ConstantInputWarning
from pyAutoSummarizer.base.evaluation.base import get_summary_evaluation
from research.src.constants import (
    PATH_SUMMEVAL_JSONL,
    LEXICAL_EVAL,
    LEXICAL_PREFIX,
    SEMANTIC_PREFIX,
    SEMANTIC_EVAL,
    JOIN_COLS,
    LEXICAL_COL,
    SEMANTIC_COL,
    HIBRITY_QUALITY_SCORE,
    EVAL_COLS,
    METHODS,
    N,
    HUMAN_COLS,
    FINAL_METRIC
)


warnings.filterwarnings("ignore", message="IProgress not found.*")
warnings.filterwarnings("ignore", category=ConstantInputWarning)
tqdm.pandas()


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

def get_combinated_metric(avg_summeval_metrics, df_agg, join_cols, A1=None, A2=None,B1=None,B2=None):

    combined_df = pd.merge(avg_summeval_metrics, df_agg, on=join_cols, how='inner')

    # 1) Identifica as colunas que vamos ponderar
    lexical_cols = [
        c for c in combined_df.columns
        if c.startswith(f"{LEXICAL_PREFIX}_") and not c.endswith("_overall_mean")
    ]
    semantic_cols = [
        c for c in combined_df.columns
        if c.startswith(f"{SEMANTIC_PREFIX}_") and not c.endswith("_overall_mean")
    ]

    # 2) Inicializa A1 e A2 com pesos iguais, se não fornecidos
    n_lex = len(lexical_cols)
    n_sem = len(semantic_cols)
    if A1 is None:
        A1 = [1.0 / n_lex] * n_lex
    if A2 is None:
        A2 = [1.0 / n_sem] * n_sem

    # 3) Verifica tamanho dos vetores
    if len(A1) != n_lex or len(A2) != n_sem:
        raise ValueError(f"A1 deve ter {n_lex} elementos e A2 deve ter {n_sem} elementos.")

    # 4) Calcula soma ponderada
    #    – lexical_score: soma A1_i * coluna_lexical_i
    #    – semantic_score: soma A2_j * coluna_semantic_j
    lexical_score = sum(
        weight * combined_df[col]
        for weight, col in zip(A1, lexical_cols)
    )
    semantic_score = sum(
        weight * combined_df[col]
        for weight, col in zip(A2, semantic_cols)
    )

    if B1 is None:
        B1 = .5
        B2 = .5
    soma_B = B1 + B2
    B1 = soma_B/B1
    B2 = soma_B/B2
    
    combined_df[HIBRITY_QUALITY_SCORE] = B1*lexical_score + B2*semantic_score

    return combined_df

def get_agg_frame(n=N):
    df = pd.read_json(PATH_SUMMEVAL_JSONL, lines=True)
    avg_summeval_metrics = get_metrics_annotations(df)
    df_sample = df.sample(N).copy()
    df_agg_lexical = get_metrics_evaluator(df_sample,LEXICAL_EVAL , LEXICAL_PREFIX)
    df_agg_semantic= get_metrics_evaluator(df_sample,SEMANTIC_EVAL , SEMANTIC_PREFIX )
    df_agg = pd.merge(df_agg_lexical, df_agg_semantic, on= JOIN_COLS)
    return (avg_summeval_metrics,df_agg)

def get_final_corr(avg_summeval_metrics,df_agg,A1=None,A2=None,B1=None,B2=None):
    metrics_frame = get_combinated_metric(avg_summeval_metrics, df_agg, JOIN_COLS,A1,A2)
    correlation_table = get_corr(metrics_frame, EVAL_COLS, HUMAN_COLS,  METHODS)
    FINAL_CORR = correlation_table.loc[HIBRITY_QUALITY_SCORE,FINAL_METRIC]
    return FINAL_CORR

def get_metric_columns(df):
    """
    Retorna duas listas:
     - lexical_cols: todas as colunas que começam com LEXICAL_PREFIX_ (exceto _overall_mean)
     - semantic_cols: todas as colunas que começam com SEMANTIC_PREFIX_ (exceto _overall_mean)
    """
    lexical_cols = [
        c for c in df.columns
        if c.startswith(f"{LEXICAL_PREFIX}_") and not c.endswith("_overall_mean")
    ]
    semantic_cols = [
        c for c in df.columns
        if c.startswith(f"{SEMANTIC_PREFIX}_") and not c.endswith("_overall_mean")
    ]
    return lexical_cols, semantic_cols

