import pandas as pd
from research.src.functions import get_metric_columns
from research.src.functions import get_final_corr

def objective(trial,avg_summeval_metrics, df_agg):
    # faz o merge interno para descobrir colunas disponíveis
    lexical_cols, semantic_cols = get_metric_columns(df_agg)

    n_lex = len(lexical_cols)
    n_sem = len(semantic_cols)
    # sugere um peso para cada coluna lexical e semantic
    A1 = [trial.suggest_float(f"A1_{i}", 0.0, 1.0) for i in range(n_lex)]
    A2 = [trial.suggest_float(f"A2_{j}", 0.0, 1.0) for j in range(n_sem)]

    # normaliza para soma = 1 (ou usa distribuição uniforme se tudo zero)
    sum1 = sum(A1)
    if sum1 > 0:
        A1 = [a / sum1 for a in A1]
    else:
        A1 = [1.0 / n_lex] * n_lex

    sum2 = sum(A2)
    if sum2 > 0:
        A2 = [a / sum2 for a in A2]
    else:
        A2 = [1.0 / n_sem] * n_sem

    try:
        # get_final_corr agora espera A1 e A2 como listas de pesos
        corr = get_final_corr(avg_summeval_metrics, df_agg, A1, A2)
        return corr if pd.notnull(corr) else float('-inf')
    except Exception as e:
        print(f"Erro ao calcular corr(A1={A1}, A2={A2}): {e}")
        return float('-inf')
    
def get_best_weights(study, df_agg):
    best_params = study.best_params
    best_value  = study.best_value

    lexical_cols, semantic_cols = get_metric_columns(
        df_agg
    )

    n_lex = len(lexical_cols)
    n_sem = len(semantic_cols)

    # 2) Extrai e normaliza os pesos sugeridos
    A1_raw = [best_params[f"A1_{i}"] for i in range(n_lex)]
    A2_raw = [best_params[f"A2_{j}"] for j in range(n_sem)]

    sum1 = sum(A1_raw) or 1.0
    sum2 = sum(A2_raw) or 1.0

    A1 = [w / sum1 for w in A1_raw]
    A2 = [w / sum2 for w in A2_raw]

    print("\n🧪 Resultado do Otimização de Pesos")
    print("───────────────────────────────────")
    print("▶️ Pesos Lexicais (A1):")
    for col, w in zip(lexical_cols, A1):
        print(f"    • {col:<20}: {w:.4f}")

    print("\n▶️ Pesos Semânticos (A2):")
    for col, w in zip(semantic_cols, A2):
        print(f"    • {col:<20}: {w:.4f}")

    print("\n▶️ Melhor correlação (objetivo):")
    print(f"    {best_value:.4f}\n")