from torch.optim import Adam
from torch.nn import Module


def build_optimizer(model: Module, optim_params: dict):
    type = optim_params.get('type')

    match type:
        case "Adam":
            return Adam(model.parameters(), **optim_params.get('params'))
        case _:
            raise ValueError(
                f"Unsupported optimizer: {type}"
            )