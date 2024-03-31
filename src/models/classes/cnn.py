from torch.nn import Module, Sequential


class SpectrogramCNN(Module):
    def __init__(self, architecture: Sequential) -> None:
        super(SpectrogramCNN, self).__init__()
        self.arch = architecture
    
    def forward(self, x):
        return self.arch(x)