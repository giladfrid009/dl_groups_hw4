from typing import Iterable, Callable, Iterator
from collections import deque
import torch
from torch import nn, Tensor

from src.permutation import Permutation


class CanonicalModel(nn.Module):
    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        x = torch.sort(x, dim=1, descending=True).values
        return self.model(x)


class SymmetryModel(nn.Module):
    def __init__(
        self,
        model: nn.Module,
        perm_creator: Callable[[None], Iterator[Permutation]],
        chunksize: int = 1,
    ) -> None:
        super().__init__()
        self.model = model
        self.perm_creator = perm_creator
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
        total = 0
        result = None

        perms = self.perm_creator()
        for perm_chunk in self._chunk(perms, self.chunksize):
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
    tolerance: float = 1e-3,
) -> bool:

    assert input.ndim == 3, "Input must be of shape (batch, channels, features)"

    if device is None:
        device = model.parameters().__next__().device

    model = model.to(device)
    input = input.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(test_rounds):
            perm = Permutation(torch.randperm(input.shape[1]))

            out1 = model(perm(input))
            out2 = model(input)

            if not torch.allclose(out1, out2, atol=tolerance):
                return False

    return True


def test_equivariant(
    model: nn.Module,
    input: Tensor,
    device: torch.device | None = None,
    test_rounds: int = 5,
    tolerance: float = 1e-3,
) -> bool:

    assert input.ndim == 3, "Input must be of shape (batch, channels, features)"

    if device is None:
        device = model.parameters().__next__().device

    model = model.to(device)
    input = input.to(device)

    model.eval()
    with torch.no_grad():
        for _ in range(test_rounds):
            perm = Permutation(torch.randperm(input.shape[1]))

            out1 = model(perm(input))
            out2 = perm(model(input))

            if not torch.allclose(out1, out2, atol=tolerance):
                return False

    return True
