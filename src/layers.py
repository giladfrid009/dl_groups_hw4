import math
import torch
from torch import Tensor
from torch import nn


class LinearEquivariant(nn.Module):
    __constants__ = ["in_channels", "out_channels"]

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initializes a LinearEquivariant module.
        This module is a custom linear layer that is equivariant to permutations of the input.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.randn(in_channels, out_channels))
        self.alpha = nn.Parameter(torch.randn(in_channels, out_channels))
        self.beta = nn.Parameter(torch.randn(in_channels, out_channels))

        range = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.alpha, -range, range)
        nn.init.uniform_(self.beta, -range, range)
        nn.init.uniform_(self.bias, -range, range)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs linear equivariant transformation on the input tensor.

        Data dimensions:
        * B is batch size
        * N is sequence length

        Args:
            x (Tensor): Input tensor of shape (B, N, in_channels).

        Returns:
            Tensor: Output tensor of shape (B, N, out_channels).
        """

        assert x.ndim == 3
        assert x.shape[-1] == self.in_channels

        # shape (B, N, in_channels, 1)
        x = x.unsqueeze(-1)

        # shape (B, 1, in_channels, 1)
        x_sum = torch.sum(x, dim=1, keepdim=True)

        # shape (B, N, in_channels, out_channels)
        all = x * self.alpha + x_sum * self.beta + self.bias

        # shape (B, N, out_channels)
        reduced = torch.sum(all, dim=2)

        return reduced


class LinearInvariant(nn.Module):
    __constants__ = ["in_channels", "out_channels"]

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Initialize the LinearInvariant module.
        This module is a custom linear layer that is invariant to permutations of the input.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bias = nn.Parameter(torch.randn(in_channels, out_channels))
        self.alpha = nn.Parameter(torch.randn(in_channels, out_channels))

        range = 1 / math.sqrt(in_channels)
        nn.init.uniform_(self.alpha, -range, range)
        nn.init.uniform_(self.bias, -range, range)

    def forward(self, x: Tensor) -> Tensor:
        """
        Performs linear invariant transformation on the input tensor.

        Data dimensions:
        * B is batch size
        * N is sequence length

        Args:
            x (Tensor): Input tensor of shape (B, N, in_channels).

        Returns:
            Tensor: Output tensor of shape (B, 1, out_channels).
        """

        assert x.ndim == 3
        assert x.shape[-1] == self.in_channels

        # shape (B, N, in_channels, 1)
        x = x.unsqueeze(-1)

        # shape (B, 1, in_channels, 1)
        x_sum = torch.sum(x, dim=1, keepdim=True)

        # shape (B, 1, in_channels, out_channels)
        all = x_sum * self.alpha + self.bias

        # shape (B, 1, out_channels)
        reduced = torch.sum(all, dim=2)

        return reduced


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)

        if d_model % 2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:, 0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        # register_buffer => Tensor which is not a parameter, but should be part of the modules state.
        # Used for tensors that need to be on the same device as the module.
        # persistent=False tells PyTorch to not add the buffer to the state dict (e.g. when we save the model)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Applies positional encoding to the input tensor.

        Data dimensions:
        * B is batch size
        * N is sequence length
        * D is feature / channel dimension

        Args:
            x (Tensor): Input tensor of shape (B, N, D).

        Returns:
            Tensor: Output tensor of shape (B, N, D) with positional encoding applied.
        """

        x = x + self.pe[:, : x.size(1), :]
        return x
