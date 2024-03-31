import torch


from tqdm import tqdm

from torch.utils.data import DataLoader
from src.utils.saving.model import export_model_onnx, build_checkpoint, save_model_in_pt, save_model_in_pt_jit
from src.metrics import build_dict_metrics, build_dict_metrics_values, calc_metrics_values
from src.utils.data import append_list_value_in_dict, extend_list_value_in_dict
from src.utils.saving.data import save_metrcis


class API:
    def __init__(
            self, 
            model: torch.nn.Module, 
            criterion: torch.nn.Module,
            optimizer: torch.optim.Optimizer=None,
            device: torch.device=torch.device("cpu"), 
            ):
        """
        Initialize the API class with the given model, loss function, optimizer, and device.

        Args:
            model (torch.nn.Module): PyTorch model to train.
            criterion (torch.nn.Module): Loss function to use for training.
            optimizer (torch.optim.Optimizer): Optimizer to be used for training.
            device (torch.device): Device to move the model and data to for training.
        """

        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train_step(self, input_data: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Perform a single training step with the provided input_data and labels.

        Args:
            input_data (torch.Tensor): Input data tensor.
            labels (torch.Tensor): Target data tensor.
        """

        # Set model to training mode
        self.model.train()  

        # Reset gradients
        self.optimizer.zero_grad()

        # Move inputs and labels to the device
        input_data, labels = input_data.to(self.device), labels.to(self.device)

        # Forward pass, compute the output
        outputs = self.model(input_data)

        # print(input_data.shape)

        # Calculate loss
        loss = self.criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Update the model weights
        self.optimizer.step()

        # Calculate metrics
        _, predictions = torch.max(outputs.data, dim=1)

        return labels.tolist(), predictions.tolist(), loss.item()
    
    def evaluate_step(self, input_data: torch.Tensor, labels: torch.Tensor) -> dict:
        """
        Perform a single evaluation step with the provided inputs and labels.
        This method does not update the model weights; it is used for validation and testing.


        Args:
            inputs (torch.Tensor):Input data tensor.
            labels (torch.Tensor): Target data tensor.
        """

        # Set model to evaluation mode
        self.model.eval()

        with torch.no_grad(): 
            # Move input_data and labels to the device
            input_data, labels = input_data.to(self.device), labels.to(self.device)

            # Forward pass, compute the output
            outputs = self.model(input_data)

            # Calculate loss
            loss = self.criterion(outputs, labels)

            # Calculate metrics
            _, predictions = torch.max(outputs.data, dim=1)

        return labels.tolist(), predictions.tolist(), loss.item()
    
    def fit(
            self, 
            data: torch.utils.data.Dataset,
            batch_size: int, 
            shuffle: bool = False,
            num_workers: int = 0,
            valid_data: torch.utils.data.Dataset = None,
            validation_batch_size: int = 4, 
            validation_freq: int = 1, 
            epochs: int = 1, 
            initial_epoch: int = 0, 
            saving_onnx_params: dict = None,
            saving_pt_params: dict = None,
            saving_jit_params: dict = None,
            saving_checkpoint_params: dict = None,
            ):
        y_true, y_pred, loss = [], [], []

        train_dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers, shuffle=shuffle)
        
        if valid_data is not None:
            valid_dataloader = DataLoader(valid_data, batch_size=validation_batch_size, num_workers=num_workers)

        for epoch in tqdm(range(initial_epoch, epochs)):
            y_true_epoch, y_pred_epoch, loss_epoch = [], [], []

            if valid_data is not None:
                valid_y_true_epoch, valid_y_pred_epoch, valid_loss_epoch = [], [], []
            
            for data, target in train_dataloader:
                ...

            if valid_data is not None and (epoch - initial_epoch) % validation_freq == 0:  
                for data, target in valid_dataloader:
                    ...

            if saving_checkpoint_params is not None:
                if valid_data is not None:
                    build_checkpoint(epoch, 
                                    self.model, 
                                    self.optimizer,
                                    epoch_metrics_values, # y_true, y_sroce
                                    saving_checkpoint_params.get('path'),
                                    f"{saving_checkpoint_params.get('name')}_{epoch}",
                                    valid_epoch_metrics_values # valid y_true, y_sroce
                                    )
                else:
                    build_checkpoint(epoch, 
                                    self.model,
                                    self.optimizer,
                                    epoch_metrics_values,  # y_true, y_sroce
                                    saving_checkpoint_params.get('path'),
                                    f"{saving_checkpoint_params.get('name')}_{epoch}"
                                    )

        if saving_onnx_params is not None:
            export_model_onnx(self.model, **saving_onnx_params)

        if saving_pt_params is not None:
            save_model_in_pt(self.model, **saving_pt_params)

        if saving_jit_params is not None:
            save_model_in_pt_jit(self.model, **saving_pt_params)


    def evaluate(self, 
                data, 
                batch_size: int, 
                num_workers: int = 0,
                saving_metrics_params: dict = None, 
                ):
        metrics_values = build_dict_metrics_values(self.metrics)
        metrics_values["loss"] = []

        dataloader = DataLoader(data, batch_size=batch_size, num_workers=num_workers)

        with torch.no_grad():   
            for batch_idx, (data, target) in enumerate(dataloader):
               metrics_values = extend_list_value_in_dict(metrics_values, self.evaluate_step(data, target))
        
        if saving_metrics_params is not None:
            save_metrcis(metrics_values, **saving_metrics_params)
