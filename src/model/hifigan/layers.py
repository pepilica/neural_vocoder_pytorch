from typing import Tuple

from torch import nn


class WNormConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | Tuple[int], 
                 stride: int | Tuple[int] = 1, padding: str | int | Tuple[int] = 0, 
                 dilation: int | Tuple[int] = 1, groups: int = 1, bias: bool = True, 
                 padding_mode: str = "zeros", device=None, dtype=None, init_norm=True) -> None:
        super().__init__()
        self.layer = nn.utils.weight_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, 
                      dilation, groups, bias, padding_mode, device, dtype)
        )
    
    def forward(self, x):
        return self.layer(x)

    @property
    def weight(self):
        return self.layer.weight
    

class WNormConvTranspose1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | Tuple[int], 
                 stride: int | Tuple[int] = 1, padding: str | int | Tuple[int] = 0) -> None:
        super().__init__()
        self.layer = nn.utils.weight_norm(
            nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding)
        )
    
    def forward(self, x):
        return self.layer(x)

    @property
    def weight(self):
        return self.layer.weight


class WNormConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | Tuple[int], 
                 stride: int | Tuple[int] = 1, padding: str | int | Tuple[int] = 0, 
                 dilation: int | Tuple[int] = 1, groups: int = 1, bias: bool = True, 
                 padding_mode: str = "zeros", device=None, dtype=None) -> None:
        super().__init__()
        self.layer = nn.utils.weight_norm(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, 
                      dilation, groups, bias, padding_mode, device, dtype)
        )

    def forward(self, x):
        return self.layer(x)

    @property
    def weight(self):
        return self.layer.weight
    

class SNormConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int | Tuple[int], 
                 stride: int | Tuple[int] = 1, padding: str | int | Tuple[int] = 0, 
                 dilation: int | Tuple[int] = 1, groups: int = 1, bias: bool = True, 
                 padding_mode: str = "zeros", device=None, dtype=None) -> None:
        super().__init__()
        self.layer = nn.utils.spectral_norm(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding, 
                      dilation, groups, bias, padding_mode, device, dtype)
        )
    
    def forward(self, x):
        return self.layer(x)

    @property
    def weight(self):
        return self.layer.weight
