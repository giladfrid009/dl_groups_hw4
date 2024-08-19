import itertools
from collections import deque
from typing import Iterator
import torch
from torch import Tensor
from torch import nn


class Permutation(nn.Module):
    def __init__(self, perm: torch.Tensor) -> None:
        super().__init__()

        perm = perm.detach().clone()
        self.register_buffer("perm", perm, persistent=False)
        self.hash = hash(tuple(perm.tolist()))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            return x[self.perm]
        return x[:, self.perm]

    def __len__(self) -> int:
        return len(self.perm)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Permutation):
            return torch.equal(self.perm, other.perm)
        return False

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __hash__(self) -> int:
        return self.hash


class RandomPermute(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """
        Randomly permute the input tensor along the sequence dimension.

        Data dimensions:
        * B is batch size
        * N is sequence length
        * D is feature / channel dimension

        Args:
            x (Tensor): Input tensor of shape (B, N, D).

        Returns:
            Tensor: If training, returns a randomly permuted tensor of shape (B, N, D) along dimension 1.
                Otherwise, returns the input tensor.
        """
        if self.training is False:
            return x

        perm = torch.randperm(x.shape[1])
        return x[:, perm]


def create_all_permutations(perm_length: int) -> Iterator[Permutation]:
    for perm in itertools.permutations(range(perm_length)):
        yield Permutation(torch.tensor(perm, dtype=torch.int32))


def create_permutations_from_generators(generators: list[Permutation]) -> Iterator[Permutation]:
    def compose(p1: Permutation, p2: Permutation) -> Permutation:
        return Permutation(p1.forward(p2.perm))

    length = len(generators[0])
    id = Permutation(torch.arange(length))
    generated_perms = {id}
    queue = deque([id])

    yield id

    while queue:
        current_perm = queue.popleft()
        for gen in generators:
            new_perm = compose(current_perm, gen)
            if new_perm not in generated_perms:
                generated_perms.add(new_perm)
                queue.append(new_perm)
                yield new_perm
