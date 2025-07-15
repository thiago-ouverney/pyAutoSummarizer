class EvaluationMetric:
    def __init__(self, name):
        self.name = name

    def evaluate(self, reference: str, generated: str) -> dict:
        raise NotImplementedError("You must implement the evaluate method")


def get_summary_evaluation(reference_summary, generated_summary, evaluators=[]):
    """
    Avalia um resumo gerado em relação ao de referência usando métricas fornecidas.

    Args:
        reference_summary (str): O resumo considerado como referência.
        generated_summary (str): O resumo gerado a ser avaliado.
        evaluators (list): Lista de instâncias de métricas (EvaluationMetric).

    Returns:
        dict: Resultados da avaliação.
    """
    if not evaluators:
        print("Nenhuma métrica fornecida para avaliação.")
        return {}

    results = {}
    for evaluator in evaluators:
        results.update(evaluator.evaluate(reference_summary, generated_summary))
    return results