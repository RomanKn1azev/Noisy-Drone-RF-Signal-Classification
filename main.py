import warnings
warnings.filterwarnings("ignore")


from argparse import ArgumentParser

from src.training import train
from src.testing import test
from src.metrics import visualize_metrics
from src.create_datasets import creating_datasets

from src.utils.file import load_yml_file

from src.data.dataset import DroneSignalsDatasetSpec
from src.utils.saving.data import save_dataset
from src.utils.visualization.metrics import visualize_multiclass_accuracy
from src.utils.loading.data import load_torch_data


def run_task(type: str, cfg_path: str):
    params = load_yml_file(cfg_path)

    match type:
        case "create_datasets":
            creating_datasets(params)

        case "train":
            train(params)

        case "test":
            test(params)

        case "metrics":
            visualize_metrics(params)

        case _:
            raise ValueError(f"Unknown task type '{type}'")
        

def main():
    parser = ArgumentParser()

    parser.add_argument('-mode', type=str, required=True, help='Type of task')

    parser.add_argument('-cfg', type=str, required=True, help='Path to config file.')

    args = parser.parse_args()
    
    mode = args.mode
    cfg_path = args.cfg

    run_task(mode, cfg_path)


if __name__ == "__main__":
    main()
    # path = "saved_objects/test/models/checkpoints/pinpoint.pt"

    # data = load_torch_data(path)

    # print(len(data['metrics']['accuracy']))

    # visualize_multiclass_accuracy((data['metrics']['accuracy']))


    # path = "data/processed/spec.pt"
    # dataset = load_torch_data(path)

    # # define a data loader
    # data_loader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=64)

    # print('Dataset loaded to torch data_loader.')

    # # get a batch of samples from the data loader
    # x_spec, labels = next(iter(data_loader))

    # print('Loaded a batch of samples from the data loader with batch size', x_spec.shape[0], 'and the following shapes:')
    # print('x_spec shape: ', x_spec.shape)
    # print('labels shape: ', labels.shape)
