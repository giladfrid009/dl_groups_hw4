import math
import torch
from torch import Tensor
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        device: torch.device | None = None,
        var1: float = 1.0,
        var2: float = 0.8,
        static: bool = True,
    ) -> None:
        super().__init__()

        if device is None:
            device = torch.device("cpu")

        self.shape = shape
        self.device = device
        self.var1 = var1
        self.var2 = var2
        self.static = static

        labels = torch.randint(0, 2, (num_samples,)).to(torch.float32)
        self.labels = labels.to(device)

        if static:
            var = var1 * (self.labels == 0) + var2 * (self.labels == 1)
            var = var.view(num_samples, *([1] * len(shape)))
            data = torch.randn(size=(num_samples, *shape)) * torch.sqrt(var)
            self.data = data.to(device)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        label = self.labels[idx]

        if not self.static:
            var = self.var1 if label == 0 else self.var2
            data = torch.randn(self.shape) * math.sqrt(var)
            data = data.to(self.device)
        else:
            data = self.data[idx]

        return data, label
