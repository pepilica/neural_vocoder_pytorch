from torch import nn
from torch.nn import functional as F
from src.model.hifigan.layers import WNormConv2d


class MPDDiscriminator(nn.Module):
    def __init__(self, period, channels, final_channels=1024, block_kernel=(5, 1), 
                 block_stride=(3, 1), b4_last_kernel=(5, 1), b4_last_padding=(2, 0), 
                 last_kernel=(5, 1), last_padding=(2, 0), relu_scope=0.1) -> None:
        super().__init__()
        self.period = period
        channels = [1] + channels
        self.blocks = nn.ModuleList([
            nn.Sequential(
                WNormConv2d(
                    channels[i],
                    channels[i + 1],
                    block_kernel,
                    block_stride,
                    ((block_kernel[0] - 1) // 2, 0)
                ),
                nn.LeakyReLU(relu_scope)
            ) for i in range(len(channels) - 1)
        ])
        self.blocks.append(nn.Sequential(
            WNormConv2d(channels[-1], 
                      final_channels,
                      b4_last_kernel,
                      padding=b4_last_padding),
            nn.LeakyReLU(relu_scope)
        ))
        self.last_layer = nn.Conv2d(
            final_channels,
            1,
            last_kernel,
            padding=last_padding,
        )

    def forward(self, x):
        if (s := x.shape[-1] % self.period) != 0:
            x = F.pad(x, (0, self.period - s), mode="reflect")
        B, S = x.shape
        x = x.reshape((B, 1, S // self.period, self.period)).contiguous()

        feature_maps = []
        for layer in self.blocks:
            x = layer(x)
            feature_maps.append(x)
        x = self.last_layer(x)
        return x.flatten(1, -1), feature_maps


class MPD(nn.Module):
    def __init__(self,
                 periods,
                 block_kernel,
                 block_stride,
                 channels,
                 **kwargs):
        super().__init__()
        self.discriminators = nn.ModuleList([
            MPDDiscriminator(
                period=period,
                block_kernel=block_kernel,
                block_stride=block_stride,
                channels=channels
            )
            for period in periods
        ])
    
    def forward(self, audio, **batch):
        outputs = []
        features = []
        for base_discriminator in self.discriminators:
            output, features_list = base_discriminator(audio)
            outputs.append(output)
            features.append(features_list)
        return outputs, features

    def __str__(self) -> str:
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = f"All parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info