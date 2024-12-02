from typing import Tuple

from torch import nn


class WNormConv1d(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.layer = nn.utils.weight_norm(
            nn.Conv1d(*args, **kwargs)
        )
    
    def forward(self, x):
        return self.layer(x)
    

class WNormConvTranspose1d(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.layer = nn.utils.weight_norm(
            nn.ConvTranspose1d(*args, **kwargs)
        )
    
    def forward(self, x):
        return self.layer(x)


class WNormConv2d(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.layer = nn.utils.weight_norm(
            nn.Conv2d(*args, **kwargs)
        )

    def forward(self, x):
        return self.layer(x)
    

class SNormConv1d(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.layer = nn.utils.spectral_norm(
            nn.Conv1d(*args, **kwargs)
        )
    
    def forward(self, x):
        return self.layer(x)
