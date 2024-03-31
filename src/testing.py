from src.utils.setting import setting_device, setting_seed
from src.utils.loading.data import load_dataset, load_torch_data
from src.criterion import build_criterion
from src.utils.file import load_yml_file
from src.utils.loading.model import load_jit_model

from src.learning.supervised.classification.neural_networks.model import API


def test(params: dict):
    setting_seed(params["random_state"])

    data = load_dataset(params['data']['path'])

    device = setting_device(params.get('device', 'cpu'))
    
    metrics = params.get("metrics", None)

    criterion = build_criterion(params['criterion'])
    
    model = load_jit_model(params["model"]["path"])

    api = API(
        model,
        criterion,
        device=device,
        metrics=metrics
    )

    api.evaluate(
        data=data,
        **params['evaluate']
    )