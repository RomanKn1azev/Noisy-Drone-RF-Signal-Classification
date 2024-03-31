from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, SoftMarginLoss


def build_criterion(criterion: dict):
    type = criterion.get('type') 

    match type:
        case "CrossEntropyLoss":
            return CrossEntropyLoss()
        case "BCEWithLogitsLoss":       
            return BCEWithLogitsLoss() # в конце добавляется слой в виде сигмоиды
        case "SoftMarginLoss":
            return SoftMarginLoss() # метки должны иметь вид 1 и -1.
        case _:
            raise ValueError(
                f"Unsupported loss function: {type}"
            )