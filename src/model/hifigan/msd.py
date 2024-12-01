import math

from torch import nn
from torch.nn import functional as F
from src.model.hifigan.layers import WNormConv1d, SNormConv1d


class MSDDiscriminator(nn.Module):
    def __init__(self, pooling_factor, channels, kernels, strides, groups, relu_slope=0.1) -> None:
        super().__init__()
        self.pooling_factor = pooling_factor
        if pooling_factor == 1:
            self.pooling = nn.Identity()
            conv_type = SNormConv1d
        else:
            self.pooling = nn.Sequential(
                *[nn.AvgPool1d(kernel_size=4, stride=2, padding=2) for _ in range(int(math.log2(pooling_factor)))]
            )
            conv_type = WNormConv1d
        channels = [1] + channels
        self.blocks = nn.ModuleList([
            nn.Sequential(
                conv_type(channels[i],
                          channels[i + 1],
                          kernels[i],
                          strides[i],
                          (kernels[i] - 1) // 2,
                          groups=groups[i]
                          ),
                nn.LeakyReLU(relu_slope)
            )
            for i in range(len(channels) - 1)
        ])
        self.head = conv_type(
            channels[-1],
            1,
            kernel_size=3,
            stride=1,
            padding=1
        )

    def forward(self, x):
        feature_maps = []
        x = self.pooling(x)
        if len(x.shape) < 3:
            x = x.unsqueeze(1)
        for layer in self.blocks:
            x = layer(x)
            feature_maps.append(x)
        x = self.head(x)
        return x, feature_maps
    

class MSD(nn.Module):
    def __init__(self,
                 pooling_factors,
                 kernels,
                 strides,
                 groups,
                 channels):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MSDDiscriminator(
                pooling_factor,
                channels, 
                kernels,
                strides, 
                groups
            )
            for pooling_factor in pooling_factors
        ])
    
    def forward(self, audio, **batch):
        outputs = []
        features = []
        for base_discriminator in self.discriminators:
            output, features_list = base_discriminator(audio)
            outputs.append(output)
            features.extend(features_list)
        return outputs, features
