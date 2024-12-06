from torch import nn
from torch.nn import functional as F

from src.model.hifigan.layers import WNormConv1d, WNormConvTranspose1d
from src.utils.audio_utils import MelSpectrogramConfig


class ResBlock(nn.Module):
    def __init__(self, dilations, kernel, channels, relu_slope=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(*[
                nn.Sequential(
                    nn.LeakyReLU(negative_slope=relu_slope),
                    WNormConv1d(channels, channels, kernel, dilation=dilations[j][i], padding="same")
                ) for i in range(len(dilations[0]))])
            for j in range(len(dilations))
        ])

    def forward(self, x):
        cur_x = x
        for block_i in self.layers:
            cur_x = cur_x + block_i(cur_x)
        return x
    

class MultiReceptiveFieldFusion(nn.Module):
    def __init__(self, dilations, kernels, channels, relu_slope=0.1) -> None:
        super().__init__()
        self.layers = nn.ModuleList((
            ResBlock(dilations[i], kernels[i], channels, relu_slope) for i in range(len(dilations))
        ))

    def forward(self, x):
        result = None 
        for block_i in self.layers:
            output_i = block_i(x)
            result = result + output_i if result is not None else output_i
        return result / len(self.layers)


class Generator(nn.Module):
    def __init__(self, kernels_mrf, kernels_upsample, dilations, encoder_channels, 
                 kernel_encoder=7, dilation_encoder=1, kernel_head=7, relu_slope=0.1):
        super().__init__()
        self.encoder = WNormConv1d(MelSpectrogramConfig.n_mels, encoder_channels, 
                                   kernel_size=kernel_encoder, dilation=dilation_encoder, padding="same")
        cur_num_channels = encoder_channels
        upsampling_layers = []
        for i in range(len(kernels_upsample)):
            upsampling_layers.append(nn.Sequential(
                nn.LeakyReLU(relu_slope),
                WNormConvTranspose1d(cur_num_channels, cur_num_channels // 2, 
                                     kernels_upsample[i], kernels_upsample[i] // 2,
                                     kernels_upsample[i] // 4),
                MultiReceptiveFieldFusion(dilations, kernels_mrf, cur_num_channels // 2, relu_slope)
            ))
            cur_num_channels //= 2
        self.upsampling_layers = nn.Sequential(*upsampling_layers)
        self.audio_head = nn.Sequential(
            nn.LeakyReLU(relu_slope),
            WNormConv1d(cur_num_channels, 1, kernel_head, padding='same'),
            nn.Tanh()
        )

    def forward(self, mel_spec, **batch):
        spec_encoded = self.encoder(mel_spec)
        spec_upsampled = self.upsampling_layers(spec_encoded)
        generated_audio = self.audio_head(spec_upsampled)
        return generated_audio.flatten(1, 2)
    
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
