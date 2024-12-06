import torch
from torch import nn
from torch.nn.functional import l1_loss

import time


class GANLossDiscriminator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, probs_gt, probs_generated, **kwargs):
        result_loss = 0
        for loss_i in map(lambda a: torch.mean((a[0] - 1) ** 2) + torch.mean(a[1] ** 2), 
                          zip(probs_gt, probs_generated)):
            result_loss = result_loss + loss_i
        return result_loss
    

class GANLossGenerator(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, probs_generated, **kwargs):
        result_loss = 0
        for output_i in probs_generated:
            result_loss = result_loss + torch.mean((output_i - 1) ** 2)
        return result_loss


class MelSpectrogramLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, spec_gt, spec_generated, **kwargs):
        return l1_loss(spec_generated, spec_gt)
    

class FeatureMatchingLoss(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, features_gt, features_generated, **kwargs):
        result_loss = 0
        for gt_i, generated_i in zip(features_gt, features_generated):
            for gt_ij, generated_ij in zip(gt_i, generated_i):
                result_loss += l1_loss(generated_ij, gt_ij)
        return result_loss
    

class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_fm=2, lambda_mel=45) -> None:
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.gan_loss_generator = GANLossGenerator()
        self.gan_loss = GANLossDiscriminator()
        self.mel_spec_loss = MelSpectrogramLoss()
        self.feature_matching_loss = FeatureMatchingLoss()

    def discriminator_loss(self, probs_gt_mpd, probs_gt_msd, 
                              probs_generated_mpd, probs_generated_msd, **batch):
        mpd_loss = self.gan_loss(probs_gt_mpd, probs_generated_mpd)
        msd_loss = self.gan_loss(probs_gt_msd, probs_generated_msd)
        return {'loss_discriminator': mpd_loss + msd_loss}

    def generator_loss(self,
                          probs_generated_mpd, probs_generated_msd, 
                          mel_spec, 
                          spec_generated, 
                          features_gt_mpd, features_gt_msd, 
                          features_generated_mpd, features_generated_msd, **batch):
        gan_loss_generator_mpd = self.gan_loss_generator(probs_generated_mpd)
        gan_loss_generator_msd = self.gan_loss_generator(probs_generated_msd)
        mel_spec_loss = self.mel_spec_loss(mel_spec, spec_generated)
        feature_matching_loss_mpd = self.feature_matching_loss(features_gt_mpd, features_generated_mpd)
        feature_matching_loss_msd = self.feature_matching_loss(features_gt_msd, features_generated_msd)
        generator_loss_term = gan_loss_generator_mpd + gan_loss_generator_msd
        mel_spec_loss_term = self.lambda_mel * mel_spec_loss
        fm_loss_term = self.lambda_fm * (feature_matching_loss_mpd + feature_matching_loss_msd)
        total_loss = generator_loss_term + mel_spec_loss_term + fm_loss_term
        return {"loss_generator": total_loss,
                'mel_spec_loss': mel_spec_loss.detach().cpu(),
                'adv_loss_generator': (gan_loss_generator_mpd + gan_loss_generator_msd).detach().cpu(),
                "feature_matching_loss": (feature_matching_loss_mpd + feature_matching_loss_msd).detach().cpu()}
