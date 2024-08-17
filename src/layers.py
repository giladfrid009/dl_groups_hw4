import math
import torch
from torch import Tensor
from torch import nn


class LinearEquivariant(nn.Module):
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
        self.bias = torch.nn.Parameter(torch.randn(in_channels, out_channels))
        self.alpha = torch.nn.Parameter(torch.randn(in_channels, out_channels))
        self.beta = torch.nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the LinearEquivariant module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, d, in_channels).

        Returns:
            Tensor: Output tensor of shape (batch_size, d, out_channels).
        """

        assert x.ndim == 3
        assert x.shape[-1] == self.in_channels

        # shape (batch_size, d, in_channels, 1)
        x = x.unsqueeze(-1)

        # shape (batch_size, 1, in_channels, 1)
        x_sum = torch.sum(x, dim=1, keepdim=True)

        # shape (batch_size, d, in_channels, out_channels)
        all = x * self.alpha + x_sum * self.beta + self.bias

        # shape (batch_size, d, out_channels)
        reduced = torch.mean(all, dim=2)

        return reduced


class LinearInvariant(nn.Module):
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
        self.bias = torch.nn.Parameter(torch.randn(in_channels, out_channels))
        self.alpha = torch.nn.Parameter(torch.randn(in_channels, out_channels))

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the LinearInvariant module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, d, in_channels).

        Returns:
            Tensor: Output tensor of shape (batch_size, 1, out_channels).
        """
        assert x.ndim == 3
        assert x.shape[-1] == self.in_channels

        # shape (batch_size, d, in_channels, 1)
        x = x.unsqueeze(-1)

        # shape (batch_size, 1, in_channels, 1)
        x_sum = torch.sum(x, dim=1, keepdim=True)

        # shape (batch_size, 1, in_channels, out_channels)
        all = x_sum * self.alpha + self.bias

        # shape (batch_size, 1, out_channels)
        reduced = torch.mean(all, dim=2)

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
        Forward pass of the PositionalEncoding module.

        Args:
            x (Tensor): Input tensor of shape (batch_size, d, in_channels).

        Returns:
            Tensor: Output tensor of shape (batch_size, d, in_channels).
        """
        x = x + self.pe[:, : x.size(1), :]
        return x
