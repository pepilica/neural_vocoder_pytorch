import itertools
import io

import torch
from torch import nn
from torch.nn import functional as F

from src.model.hifigan import generator, mpd, msd


class HiFiGAN(nn.Module):
    def __init__(self,
                 generator_params,
                 mpd_params,
                 msd_params):
        super().__init__()
        self.generator = generator.Generator(**generator_params)
        self.mpd = mpd.MPD(**mpd_params)
        self.msd = msd.MSD(**msd_params)

    def get_generator_parameters(self):
        return self.generator.parameters()
    
    def get_discriminator_parameters(self):
        return itertools.chain(self.msd.parameters(), self.mpd.parameters())
    
    def __str__(self):
        with io.StringIO() as output:
            print(super().__str__(), file=output)
            print('Generator parameters:', file=output)
            print(str(self.generator), file=output)
            print('---', file=output)
            print('MPD parameters:', file=output)
            print(str(self.mpd), file=output)
            print('---', file=output)
            print('MSD parameters:', file=output)
            print(str(self.msd), file=output)
            print('---', file=output)
            print('Overall parameters:', file=output)
            all_parameters = sum([p.numel() for p in self.parameters()])
            trainable_parameters = sum(
                [p.numel() for p in self.parameters() if p.requires_grad]
            )
            result_info = f"All parameters: {all_parameters}"
            result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

            print(result_info, file=output)
            print('---', file=output)
            return output.getvalue()