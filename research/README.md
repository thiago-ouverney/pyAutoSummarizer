# README — Pasta **`research/`**

> **Propósito**: Documentar o *workflow* de pesquisa desenvolvido para o TCC “Avaliação de Modelos Abstrativos de Sumarização: Uma Proposta de Métrica Integrada – Lexical, Semântica e Factual”.
> Este README complementa o README raiz do projeto, descrevendo **exclusivamente** o processo exploratório / experimental conduzido em *notebooks* e scripts localizados em `research/`.

---

## 📁 Estrutura de diretórios

```
research/
├── data/                  # artefatos (datasets, csv, jsonl, checkpoints…)
├── data_processing/       # scripts auxiliares de preparação de dados
│   └── pair_data.py
├── src/                   # código Python reutilizável pelos notebooks
│   ├── constants.py
│   ├── functions.py
│   └── model.py
├── eda.ipynb
├── join_data.ipynb
├── evaluators.ipynb
├── evaluator_questeval.ipynb
├── optimization.ipynb
├── summeval_metrics_notebook.ipynb
└── <outros ipynbs>
```

* **`pyAutoSummarizer/`** (nível raiz) contém as implementações genéricas de avaliadores (lexical, semântico e factual) importadas pelos notebooks.
* **`requirements.txt`** e **`Makefile`** no repositório raiz provêm o ambiente de execução (ver *Setup* abaixo).

---

## ⚙️ Setup rápido

```bash
# 1. Criar ambiente virtual (Python ≥ 3.9)
$ make create-venv      # ou python -m venv .venv

# 2. Instalar dependências e pacote em modo editável
$ make install-dependencies   # equiv. pip install -e .

# 3. (opcional) Ativar venv manualmente
$ source .venv/bin/activate   # Linux/macOS
$ .\.venv\Scripts\activate   # Windows
```

> **Dica**: Use a extensão *Jupyter* do VS Code apontando para o interpretador de `./.venv` para executar os notebooks.

---

## 🔄 Fluxo de trabalho dos notebooks

| Ordem | Notebook                              | Objetivo principal                                                                                                                             | Entradas                               | Saídas/Artefatos                           |
| ----- | ------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------------------------------------ |
| 1     | **`join_data.ipynb`**                 | Consolida o dataset **SummEval** (*model\_outputs × human annotations*) e grava `model_annotations.aligned.jsonl` em `data/`.                  | Raw jsonl do SummEval                  | `data/model_annotations.aligned.jsonl`     |
| 2     | **`eda.ipynb`**                       | Análise exploratória: distribuição de métricas humanas, inspeção de outliers, verificação de cobertura de modelos.                             | `data/model_annotations.aligned.jsonl` | gráficos/insights (inline)                 |
| 3     | **`evaluators.ipynb`**                | Calcula métricas **lexicais** (ROUGE, BLEU, METEOR), **semânticas** (BERTScore, Sentence-BERT) e **factuais** (FactCC) via `pyAutoSummarizer`. | Saída do passo 2                       | `data/df_agg.csv` (métricas agregadas)     |
| 4     | **`evaluator_questeval.ipynb`**       | Benchmark extra com **QuestEval** para comparação externa.                                                                                     | idem                                   | métricas QuestEval (csv)                   |
| 5     | **`optimization.ipynb`**              | Otimiza pesos A<sub>i</sub>/B<sub>i</sub> da métrica híbrida usando **Optuna** (`research/src/model.py`).                                      | `data/df_agg.csv` + métricas humanas   | \_best *weights.json*, curva de otimização |
| 6     | **`summeval_metrics_notebook.ipynb`** | Gera tabelas finais de correlação e validação cruzada; compila resultados para o capítulo *Resultados*.                                        | Pesos ótimos + datasets anteriores     | `data/avg_summeval_metrics.csv`, figuras   |

> **Execução incremental**: cada notebook verifica e reutiliza artefatos existentes; portanto, após rodar o passo 1 uma única vez, você pode reexecutar apenas os passos afetados por mudanças de código/dados.

---

## 📜 Scripts & módulos auxiliares

### `data_processing/pair_data.py`

Recria os pares **(artigo, resumo gerado)** a partir dos *story files* do CNN/DM quando necessário. Uso:

```bash
python research/data_processing/pair_data.py \
  --model_outputs ./data/model_outputs \
  --story_files   ./data/cnndm_stories
```

### `src/constants.py`

* Centraliza caminhos, listas de avaliadores (`LEXICAL_EVAL`, `SEMANTIC_EVAL`, `FACTUAL_EVAL`) e parâmetros globais.

### `src/functions.py`

* Funções utilitárias para:

  * agregação de métricas (`aggregate_dataframe`),
  * cálculo de correlações (`get_corr`, `get_corr_frame`),
  * construção da métrica híbrida (`get_combinated_metric`),
  * cache de artefatos (`get_agg_frame`).

### `src/model.py`

* Define **`objective()`** (função‑alvo do Optuna) e **`get_best_weights()`** para extração/normalização dos melhores pesos.

### `pyAutoSummarizer/base/evaluation/*`

* **`lexical.py`**: ROUGE, BLEU, METEOR
* **`semantic.py`**: BERTScore, Sentence‑BERT
* **`factual.py`**: FactCC
* Todas derivam da classe‑base `EvaluationMetric` (`base.py`).

---

## ▶️ Como reproduzir o experimento completo

```bash
# Dentro da pasta research/ (ambiente já criado)

jupyter lab &   # ou code . para abrir no VS Code

# 1. Execute sequencialmente os notebooks 1→6
# 2. Verifique em cada notebook a célula "Config" – ajuste paths se necessário
# 3. O resultado final (pesos + correlações) será salvo em research/data/
```

> 📫 Dúvidas ou sugestões? Abra uma *issue* ou contate **@thiago‑ouverney**.
