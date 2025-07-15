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
    FACTUAL_PREFIX,
    FACTUAL_EVAL,
    JOIN_COLS,
    HIBRITY_QUALITY_SCORE,
    EVAL_COLS,
    METHODS,
    N,
    HUMAN_COLS,
    FINAL_METRIC,FILE_PATH_DF_AGG,
    FILE_PATH_AVG_SUMMEVAL_METRICS 
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

def get_combinated_metric(avg_summeval_metrics, df_agg, join_cols, A1=None, A2=None, A3= None,
                          B1=None,B2=None, B3=None):

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

    factual_cols = [
        c for c in combined_df.columns
        if c.startswith(f"{FACTUAL_PREFIX}_") and not c.endswith("_overall_mean")
    ]

    n_lex, n_sem, n_fac = map(len, (lexical_cols, semantic_cols, factual_cols))

    # --------- 2) A-pesos (dentro de cada família)  -----------------
    def default_equal_weights(n):           # evita divisão por zero
        return [1.0 / n] * n if n else []

    A1 = A1 or default_equal_weights(n_lex)
    A2 = A2 or default_equal_weights(n_sem)
    A3 = A3 or default_equal_weights(n_fac)

    # sanity-check
    if len(A1) != n_lex or len(A2) != n_sem or len(A3) != n_fac:
        raise ValueError(
            f"Tamanhos incompatíveis: "
            f"A1={len(A1)}/{n_lex}, A2={len(A2)}/{n_sem}, A3={len(A3)}/{n_fac}"
        )

    lexical_score = sum(
        weight * combined_df[col]
        for weight, col in zip(A1, lexical_cols)
    )
    semantic_score = sum(
        weight * combined_df[col]
        for weight, col in zip(A2, semantic_cols)
    )
    factual_score = sum(
        weight * combined_df[col]
        for weight, col in zip(A3, factual_cols)
    )

    if B1 is B2 is B3 is None:
        # tudo None  ➜ pesos iguais
        B1 = B2 = B3 = 1.0 / 3
    else:
        # substitui None por 0 e normaliza
        B1 = 0.0 if B1 is None else B1
        B2 = 0.0 if B2 is None else B2
        B3 = 0.0 if B3 is None else B3
        s = B1 + B2 + B3
        if s == 0:
            raise ValueError("Ao menos um dos valores de B1, B2, B3 precisa ser não-nulo.")
        B1, B2, B3 = B1 / s, B2 / s, B3 / s

    
    combined_df[HIBRITY_QUALITY_SCORE] = B1*lexical_score + B2*semantic_score + B3*factual_score

    return combined_df

def get_agg_frame(n=N,cache=True,save=False,only_cached=False,list_ids=None):
    df_agg_cached = None
    avg_summeval_metrics = None
    if cache:
        try:
            df_agg_cached = pd.read_csv(FILE_PATH_DF_AGG)
            avg_summeval_metrics = pd.read_csv(FILE_PATH_AVG_SUMMEVAL_METRICS)
            if only_cached:
                return (avg_summeval_metrics,df_agg_cached)
        except FileNotFoundError:
            pass
    df = pd.read_json(PATH_SUMMEVAL_JSONL, lines=True)
    avg_summeval_metrics = get_metrics_annotations(df)
    df["id_model_id"] = df[JOIN_COLS[0]].astype(str) + "_" + df[JOIN_COLS[1]].astype(str)
    df_aux=pd.DataFrame()
    if list_ids:
        print(len(list_ids))
        df_aux = df[df["id_model_id"].isin(list_ids)].copy()
    df_sample = df.sample(n).copy()
    print(df_sample.shape[0])
    df_sample = pd.concat([df_sample,df_aux],ignore_index=True)
    print(df_sample.shape[0])
    df_sample = df_sample.drop_duplicates(subset="id_model_id", keep="last")
    print(df_sample.shape[0])
    # Checar oq não tem previamente salvo
    if df_agg_cached is not None:
        df_agg_cached["id_model_id"] = df_agg_cached[JOIN_COLS[0]].astype(str) + "_" + df_agg_cached[JOIN_COLS[1]].astype(str)
        ids_cached = set(df_agg_cached["id_model_id"])
        df_sample = df_sample[~df_sample["id_model_id"].isin(ids_cached)]
    if df_sample.empty:
        return (avg_summeval_metrics, df_agg_cached.drop(columns="id_model_id", errors="ignore"))
    
    df_sample = df_sample.drop(columns="id_model_id", errors="ignore").copy()
    print("Get Lexical Metris")
    df_agg_lexical = get_metrics_evaluator(df_sample,LEXICAL_EVAL , LEXICAL_PREFIX)
    print("Get Semantic Metris")
    df_agg_semantic= get_metrics_evaluator(df_sample,SEMANTIC_EVAL , SEMANTIC_PREFIX )
    print("Get Factual Metris")
    df_agg_factual= get_metrics_evaluator(df_sample,FACTUAL_EVAL , FACTUAL_PREFIX )
    df_agg = pd.merge(df_agg_lexical, df_agg_semantic, on= JOIN_COLS)
    # join final
    df_agg = pd.merge(df_agg, df_agg_factual, on= JOIN_COLS)
    # Retomando os dados previamente carregados e fazendo um union all
    df_agg= pd.concat([df_agg_cached, df_agg], axis=0) if df_agg_cached is not None else df_agg 
    if save:
        df_agg.to_csv(FILE_PATH_DF_AGG, index=False)
        avg_summeval_metrics.to_csv(FILE_PATH_AVG_SUMMEVAL_METRICS, index=False)
    return (avg_summeval_metrics,df_agg)

def get_final_corr(avg_summeval_metrics,df_agg,A1=None,A2=None,B1=None,B2=None,A3=None,B3=None):
    metrics_frame = get_combinated_metric(avg_summeval_metrics, df_agg, JOIN_COLS,A1,A2,A3,B1,B2,B3)
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
    factual_cols = [
        c for c in df.columns
        if c.startswith(f"{FACTUAL_PREFIX}_") and not c.endswith("_overall_mean")
    ]
    return lexical_cols, semantic_cols, factual_cols

