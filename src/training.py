from src.utils.loading.data import load_dataset, load_torch_data
from src.utils.setting import setting_device, setting_seed
from src.criterion import build_criterion
from src.optimizer import build_optimizer
from src.models.model_builder import build_model
from src.utils.file import load_yml_file

from src.learning.supervised.classification.neural_networks.model import API


def train(params: dict):
    state_path = params.get("state_path", None)
    state = None

    if state_path is not None:
        state = load_torch_data(params["state_path"])

    setting_seed(params["random_state"])

    data = load_dataset(params['data']['path'])
    
    valid_data = None

    if params.get("valid_data", None) is not None:
        valid_data = load_dataset(params['valid_data']['path'])

    device = setting_device(params.get('device', 'cpu'))
    metrics = params.get("metrics", None)

    model = build_model(load_yml_file(params['model_params']))

    if state is not None:
        model.load_state_dict(state["model_state_dict"])

    optimizer = build_optimizer(model, params['optimizer'])

    if state is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    criterion = build_criterion(params['criterion'])

    api = API(
        model,
        criterion,
        optimizer=optimizer,
        device=device,
        metrics=metrics
    )

    api.fit(
        data=data,
        valid_data=valid_data,
        **params['fit']
    )