import itertools
from collections import deque
from typing import Iterator

import torch
from torch import Tensor
from torch import nn


class Permutation(nn.Module):
    def __init__(self, perm: torch.Tensor) -> None:
        super().__init__()
        self.perm = perm.clone().detach()
        self.hash = hash(tuple(perm.tolist()))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 1:
            return x[self.perm]
        return x[:, self.perm]

    def __len__(self) -> int:
        return len(self.perm)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Permutation):
            return torch.all(self.perm == other.perm)
        return False

    def __ne__(self, value: object) -> bool:
        return not self.__eq__(value)

    def __hash__(self) -> int:
        return self.hash


def create_all_permutations(perm_length: int) -> Iterator[Permutation]:
    for perm in itertools.permutations(range(perm_length)):
        yield Permutation(torch.tensor(perm, dtype=torch.long))


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
