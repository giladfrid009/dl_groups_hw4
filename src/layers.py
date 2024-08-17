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
