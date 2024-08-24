from typing import Iterable
import torch
from torch import nn, Tensor

from src.permutation import Permutation


class CanonicalModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        """
        Data dimensions:
        * B is batch size
        * N is sequence length
        * D is feature / channel dimension

        Args:
            x (Tensor): Input tensor of shape (B, N, D).

        Returns:
            Tensor: Output tensor of varied dimensionality, dependent on the internal model.
        """
        x = self.canonize(x)
        return self.model(x)

    def canonize(self, x: Tensor) -> Tensor:
        sort_key = x[..., 0] + x.sum(dim=-1)
        row_idx = torch.argsort(sort_key, dim=-1)
        return torch.gather(x, dim=1, index=row_idx.unsqueeze(-1).expand_as(x))


class SymmetryModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        perms: list[Permutation],
        chunksize: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.perms = nn.ModuleList(perms)
        self.chunksize = chunksize

    def _chunk(self, data: Iterable[nn.Module], chunksize: int) -> Iterable[list[nn.Module]]:
        return (data[i : i + chunksize] for i in range(0, len(data), chunksize))

    def forward(self, x: Tensor) -> Tensor:
        """
        Data dimensions:
        * B is batch size
        * N is sequence length
        * D is feature / channel dimension

        Args:
            x (Tensor): Input tensor of shape (B, N, D).

        Returns:
            Tensor: Output tensor of varied dimensionality, dependent on the internal model.
        """

        result: Tensor | None = None

        for perm_chunk in self._chunk(self.perms, self.chunksize):
            chunksize = len(perm_chunk)
            permuted = torch.vstack([perm(x) for perm in perm_chunk])

            output: Tensor = self.model.forward(permuted)
            output = output.reshape(chunksize, -1, *output.shape[1:])
            output = torch.sum(output, dim=0)

            if result is None:
                result = output
            else:
                result = result + output

        result = result / len(self.perms)

        return result


@torch.no_grad()
def test_invariant(
    model: nn.Module,
    input: Tensor,
    device: torch.device | None = None,
    test_rounds: int = 5,
    tolerance: float = 1e-5,
) -> bool:

    assert input.ndim == 3, "Input must be of shape (B, N, D)"

    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    input = input.to(device)

    model.eval()

    for _ in range(test_rounds):
        perm = Permutation(torch.randperm(input.shape[1], device=device))
        out1 = model(perm(input))
        out2 = model(input)

        if not torch.allclose(out1, out2, atol=tolerance):
            return False

    return True


@torch.no_grad()
def test_equivariant(
    model: nn.Module,
    input: Tensor,
    device: torch.device | None = None,
    test_rounds: int = 5,
    tolerance: float = 1e-5,
) -> bool:

    assert input.ndim == 3, "Input must be of shape (B, N, D)"

    if device is None:
        device = next(model.parameters()).device

    model = model.to(device)
    input = input.to(device)

    model.train(False)

    for _ in range(test_rounds):
        perm = Permutation(torch.randperm(input.shape[1], device=device))
        out1 = model(perm(input))
        out2 = perm(model(input))

        if not torch.allclose(out1, out2, atol=tolerance):
            return False

    return True
