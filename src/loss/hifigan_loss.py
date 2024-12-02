import torch
from torch import nn
from torch.nn.functional import l1_loss


class DiscriminatorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, gt_outputs, pred_outputs):
        total_loss = 0.0
        for gt_output, pred_output in zip(gt_outputs, pred_outputs):
            gt_loss = torch.mean((gt_output - 1) ** 2)
            pred_loss = torch.mean(pred_output ** 2)
            total_loss += gt_loss + pred_loss
        return total_loss

        
class GeneratorAdvLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred_outputs):
        total_loss = 0.0
        for pred_output in pred_outputs:
            pred_loss = torch.mean((pred_output - 1) ** 2)
            total_loss += pred_loss
        return total_loss


class MelSpectrogramLoss(nn.Module):
    def __init__(self, mel_spectrogram_multiplier):
        super().__init__()
        self.mel_spectrogram_multiplier = mel_spectrogram_multiplier

    def forward(self, gt_spectrogram, pred_spectrogram):
        return self.mel_spectrogram_multiplier * l1_loss(pred_spectrogram, gt_spectrogram)
    

class FeatureMatchingLoss(nn.Module):
    def __init__(self, feature_matching_multiplier):
        super().__init__()
        self.feature_matching_multiplier = feature_matching_multiplier

    def forward(self, gt_features_list, pred_features_list):
        total_loss = 0.0
        # First cycle over different discriminators
        # Second cycle over features from different levels
        for disc_gt_features, disc_pred_features in zip(gt_features_list, pred_features_list):
            for gt_features, pred_features in zip(disc_gt_features, disc_pred_features):
                total_loss += l1_loss(pred_features, gt_features)
        return self.feature_matching_multiplier * total_loss
    

class HiFiGANLoss(nn.Module):
    def __init__(self, lambda_fm=2, lambda_mel=45) -> None:
        super().__init__()
        self.lambda_fm = lambda_fm
        self.lambda_mel = lambda_mel
        self.gan_loss_generator = GeneratorAdvLoss()
        self.gan_loss = DiscriminatorAdvLoss()
        self.mel_spec_loss = MelSpectrogramLoss(lambda_mel)
        self.feature_matching_loss = FeatureMatchingLoss(lambda_fm)

    def discriminator_loss(self, probs_gt_mpd, probs_gt_msd, 
                              probs_generated_mpd, probs_generated_msd, **batch):
        mpd_loss = self.gan_loss(probs_gt_mpd, probs_generated_mpd)
        msd_loss = self.gan_loss(probs_gt_msd, probs_generated_msd)
        return {'loss_discriminator': mpd_loss + msd_loss, 
                'mpd_loss': mpd_loss.detach().cpu(), 
                'msd_loss': msd_loss.detach().cpu()}

    def generator_loss(self,
                          probs_generated_mpd, probs_generated_msd, 
                          mel_spec, 
                          spec_generated, 
                          features_gt_mpd, features_gt_msd, 
                          features_generated_mpd, features_generated_msd, **batch):
        gan_loss_generator_mpd = self.gan_loss_generator(probs_generated_mpd)
        gpu_info = torch.cuda.mem_get_info()
        print('gan_loss_generator_mpd:', gpu_info[0] / gpu_info[1])
        gan_loss_generator_msd = self.gan_loss_generator(probs_generated_msd)
        gpu_info = torch.cuda.mem_get_info()
        print('gan_loss_generator_msd:', gpu_info[0] / gpu_info[1])
        mel_spec_loss = self.mel_spec_loss(mel_spec, spec_generated)
        gpu_info = torch.cuda.mem_get_info()
        print('mel_spec_loss:', gpu_info[0] / gpu_info[1])
        feature_matching_loss_mpd = self.feature_matching_loss(features_gt_mpd, features_generated_mpd)
        gpu_info = torch.cuda.mem_get_info()
        print('feature_matching_loss_mpd:', gpu_info[0] / gpu_info[1])
        feature_matching_loss_msd = self.feature_matching_loss(features_gt_msd, features_generated_msd)
        gpu_info = torch.cuda.mem_get_info()
        print('feature_matching_loss_msd:', gpu_info[0] / gpu_info[1])
        generator_loss_term = gan_loss_generator_mpd + gan_loss_generator_msd
        gpu_info = torch.cuda.mem_get_info()
        print('generator_loss_term:', gpu_info[0] / gpu_info[1])
        mel_spec_loss_term = mel_spec_loss
        gpu_info = torch.cuda.mem_get_info()
        print('mel_spec_loss_term:', gpu_info[0] / gpu_info[1])
        fm_loss_term = (feature_matching_loss_mpd + feature_matching_loss_msd)
        gpu_info = torch.cuda.mem_get_info()
        print('fm_loss_term:', gpu_info[0] / gpu_info[1])
        total_loss = generator_loss_term + mel_spec_loss_term + fm_loss_term
        gpu_info = torch.cuda.mem_get_info()
        print('total_loss:', gpu_info[0] / gpu_info[1])
        return {"loss_generator": total_loss,
                'generator_loss_mpd': gan_loss_generator_mpd.detach().cpu(),
                'generator_loss_msd': gan_loss_generator_msd.detach().cpu(),
                'mel_spec_loss': mel_spec_loss.detach().cpu(),
                'feature_matching_loss_mpd': feature_matching_loss_mpd.detach().cpu(),
                'feature_matching_loss_msd': feature_matching_loss_msd.detach().cpu(),
                'gan_loss_generator': (gan_loss_generator_mpd + gan_loss_generator_msd).detach().cpu(),
                "feature_matching_loss": (feature_matching_loss_mpd + feature_matching_loss_msd).detach().cpu()}
