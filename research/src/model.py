import pandas as pd
import numpy as np
from research.src.functions import get_metric_columns
from research.src.functions import get_final_corr
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold


def objective(trial,avg_summeval_metrics, df_agg):
    # faz o merge interno para descobrir colunas disponíveis
    lexical_cols, semantic_cols, factual_cols = get_metric_columns(df_agg)

    n_lex = len(lexical_cols)
    n_sem = len(semantic_cols)
    n_fac = len(factual_cols)

    # sugere um peso para cada coluna lexical e semantic
    A1 = [trial.suggest_float(f"A1_{i}", 0.0, 1.0) for i in range(n_lex)]
    A2 = [trial.suggest_float(f"A2_{j}", 0.0, 1.0) for j in range(n_sem)]
    A3 = [trial.suggest_float(f"A3_{k}", 0.0, 1.0) for k in range(n_fac)]

    B1 = trial.suggest_float("B1", 0.0, 1.0)
    B2 = trial.suggest_float("B2", 0.0, 1.0)
    B3 = trial.suggest_float("B3", 0.0, 1.0)
      # ---------- Normalização -----------------
    def normalize(vec, n):
        s = sum(vec)
        return [v / s for v in vec] if s > 0 else ([1.0 / n] * n if n else [])

    A1 = normalize(A1, n_lex)
    A2 = normalize(A2, n_sem)
    A3 = normalize(A3, n_fac)

    sB = B1 + B2 + B3
    if sB == 0:
        B1 = B2 = B3 = 1.0 / 3
    else:
        B1, B2, B3 = B1 / sB, B2 / sB, B3 / sB

    # ---------- Avalia correlação ------------
    try:
        # 2) avalia por 4-fold CV ......................
        kf = KFold(n_splits=4, shuffle=True, random_state=42)
        groups = df_agg["id"].values           # garante agrupamento por artigo
        gkf = GroupKFold(n_splits=4)
        val_corrs = []
        for _, val_idx in gkf.split(df_agg):
            # 1️⃣  subset de validação
            df_val = df_agg.iloc[val_idx].copy()
            ids    = set(df_val["id"])

            # 2️⃣  métrica humana apenas desses ids
            metrics_val = avg_summeval_metrics.loc[
                avg_summeval_metrics["id"].isin(ids)
            ].copy()

            # 3️⃣  correlação para o fold
            corr = get_final_corr(
                metrics_val, df_val,
                A1=A1, A2=A2, A3=A3,
                B1=B1, B2=B2, B3=B3
            )

            if pd.notnull(corr):
                val_corrs.append(corr)          # guarda só valores válidos

    # 4️⃣  retorna média (ou -inf caso não haja correlações válidas)
        return float(np.nanmean(val_corrs)) if val_corrs else float("-inf")
    except Exception as e:
        print(f"Erro ao calcular corr(A1={A1}, A2={A2}, A3={A3}): {e}")
        return float("-inf")
    
def get_best_weights(study, df_agg):
    best_params = study.best_params
    best_value  = study.best_value

    
    lexical_cols, semantic_cols, factual_cols = get_metric_columns(df_agg)
    n_lex, n_sem, n_fac = map(len, (lexical_cols, semantic_cols, factual_cols))

    # 2) Extrai e normaliza os pesos sugeridos
    A1_raw = [best_params[f"A1_{i}"] for i in range(n_lex)]
    A2_raw = [best_params[f"A2_{j}"] for j in range(n_sem)]
    A3_raw = [best_params[f"A3_{k}"] for k in range(n_fac)]
    B1_best_param = best_params["B1"]
    B2_best_param = best_params["B2"]
    B3_best_param = best_params["B3"]
    sB = B1_best_param + B2_best_param + B3_best_param or 1.0
    B1_best, B2_best, B3_best = B1_best_param / sB, B2_best_param / sB, B3_best_param / sB

    sum1 = sum(A1_raw) or 1.0
    sum2 = sum(A2_raw) or 1.0
    sum3 = sum(A3_raw) or 1.0

    A1 = [w / sum1 for w in A1_raw]
    A2 = [w / sum2 for w in A2_raw]
    A3 = [w / sum3 for w in A3_raw]

    print("\n🧪 Resultado do Otimização de Pesos")
    print("───────────────────────────────────")
    print("▶️ Pesos Lexicais (A1):")
    for col, w in zip(lexical_cols, A1):
        print(f"    • {col:<20}: {w:.3f}")

    print("\n▶️ Pesos Semânticos (A2):")
    for col, w in zip(semantic_cols, A2):
        print(f"    • {col:<20}: {w:.3f}")

    print("\n▶️ Pesos Factuais (A3):")
    for col, w in zip(factual_cols, A3):
        print(f"    • {col:<20}: {w:.3f}")

    print("\n▶️ Pesos Globais (B1, B2 e B3):")   
    print(f"    • B1: {B1_best:.3f}")   
    print(f"    • B2: {B2_best:.3f}")   
    print(f"    • B3: {B3_best:.3f}")   

    print("\n▶️ Melhor correlação (objetivo):")
    print(f"    {best_value:.3f}\n")