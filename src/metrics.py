import torch


from torcheval.metrics.metric import Metric
from torcheval.metrics import MulticlassAccuracy, AUC, MulticlassAUROC, MulticlassConfusionMatrix, MulticlassPrecisionRecallCurve, MulticlassRecall, MulticlassF1Score


def get_torcheval_metric(name_metric: str, params: dict) -> Metric:
    match name_metric:
        case "accuracy":
            return MulticlassAccuracy(**params)
        
        case "auc":
            return AUC()

        case "auroc":
            return MulticlassAUROC(**params)
        
        case "confusion_matrix":
            return MulticlassConfusionMatrix(**params)

        case "recall": 
            return MulticlassRecall(**params)
        
        case "precision_recall_curve":
            return MulticlassPrecisionRecallCurve(**params)
        
        case "f1score":
            return MulticlassF1Score(**params)

        case _:
            raise ValueError(f"Metric not defined for {name_metric}.")



def build_dict_metrics(metrics: dict, device: torch.device) -> dict[str, Metric]:
    return {
        name: get_torcheval_metric(name, params).to(device) for name, params in metrics.items()
    }


def build_dict_metrics_values(metrics: dict) -> dict[str, list]:
    return {
        name: [] for name in metrics.keys()
    }


def calc_metrics_values(metrics: dict[str, Metric], predictions: torch.Tensor, labels: torch.Tensor) -> dict:
    """Calculate the values of each metric given a set of predictions and labels"""
    metrics_values = {}

    for name, metric in metrics.items():
        metric.update(predictions, labels) # flatten для labels
        metrics_values[name] = [metric.compute().cpu().detach().tolist()]
        metric.reset()
    
    return metrics_values


def visualize_metrics(params: dict):
    

    ...