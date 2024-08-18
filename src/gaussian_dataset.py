import math
import torch
from torch import Tensor
from torch.utils.data import Dataset


class GaussianDataset(Dataset):
    def __init__(
        self,
        num_samples: int,
        shape: tuple[int, ...],
        var1: float = 1.0,
        var2: float = 0.8,
        static: bool = True,
    ) -> None:
        super().__init__()
        self.shape = shape
        self.var1 = var1
        self.var2 = var2
        self.static = static

        self.labels = torch.randint(0, 2, (num_samples,)).to(torch.float32)

        if static:
            var = var1 * (self.labels == 0) + var2 * (self.labels == 1)
            data = torch.randn(num_samples, *shape)
            self.data = torch.swapaxes(data.swapaxes(0, -1) * torch.sqrt(var), 0, -1)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        label = self.labels[idx]

        if not self.static:
            var = self.var1 if label == 0 else self.var2
            data = torch.randn(self.shape) * math.sqrt(var)
        else:
            data = self.data[idx]

        return data, label
