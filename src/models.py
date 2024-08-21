from typing import Iterable, Callable, Iterator
from collections import deque
import torch
from torch import nn, Tensor

from src.permutation import Permutation, RandomPermute


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
        """
        Transforms the input tensor into a canonical form.

        Data dimensions:
        * B is batch size
        * N is sequence length
        * D is feature / channel dimension

        Args:
            x (Tensor): Input tensor of shape (B, N, D).

        Returns:
            Tensor: Output tensor of shape (B, N, D).
        """
        B, N, D = x.shape

        # extract data from each sequence element features
        sort_key = x[..., 0] + x.sum(dim=-1)

        # sort sequence elements by the extracted data
        row_idx = torch.argsort(sort_key, dim=-1)

        batch_idx = torch.arange(B).unsqueeze(1).broadcast_to(B, N)

        x = x[batch_idx, row_idx]

        return x


class SymmetryModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        perms: list[Permutation],
        chunksize: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.perms = perms
        self.chunksize = chunksize

    def _chunk(self, data: Iterable[Permutation], chunksize: int) -> Iterable[list[Permutation]]:
        data_iter: Iterable[Permutation] = iter(data)
        buffer: deque[Permutation] = deque()

        while True:
            try:
                buffer.append(next(data_iter))
            except StopIteration:
                break

            if len(buffer) == chunksize:
                yield list(buffer)
                buffer.clear()

        if buffer:
            yield list(buffer)

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

        total = 0
        result: Tensor | None = None

        for perm_chunk in self._chunk(self.perms, self.chunksize):
            chunksize = len(perm_chunk)
            total += chunksize
            permuted = torch.vstack([perm(x) for perm in perm_chunk])

            output: Tensor = self.model.forward(permuted)
            output = output.reshape(chunksize, output.shape[0] // chunksize, *output.shape[1:])
            output = torch.sum(output, dim=0)

            if result is None:
                result = output
            else:
                result = result + output

        result = result / total

        return result


def test_invariant(
    model: nn.Module,
    input: Tensor,
    device: torch.device | None = None,
    test_rounds: int = 5,
    tolerance: float = 1e-5,
) -> bool:

    assert input.ndim == 3, "Input must be of shape (B, N, D)"

    if device is None:
        device = model.parameters().__next__().device

    model = model.to(device)
    input = input.to(device)

    old_status = model.training
    model.train(False)

    with torch.no_grad():
        for _ in range(test_rounds):
            perm = Permutation(torch.randperm(input.shape[1]))
            out1 = model(perm(input))
            out2 = model(input)

            if not torch.allclose(out1, out2, atol=tolerance):
                model.train(old_status)
                return False

    model.train(old_status)
    return True


def test_equivariant(
    model: nn.Module,
    input: Tensor,
    device: torch.device | None = None,
    test_rounds: int = 5,
    tolerance: float = 1e-5,
) -> bool:

    assert input.ndim == 3, "Input must be of shape (B, N, D)"

    if device is None:
        device = model.parameters().__next__().device

    model = model.to(device)
    input = input.to(device)

    old_status = model.training
    model.train(False)

    with torch.no_grad():
        for _ in range(test_rounds):
            perm = Permutation(torch.randperm(input.shape[1]))
            out1 = model(perm(input))
            out2 = perm(model(input))

            if not torch.allclose(out1, out2, atol=tolerance):
                model.train(old_status)
                return False

    model.train(old_status)
    return True
