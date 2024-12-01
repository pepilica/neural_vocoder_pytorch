import itertools

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