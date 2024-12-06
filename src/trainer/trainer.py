from pathlib import Path
import time

import pandas as pd
import torch

from src.metrics.tracker import MetricTracker
from src.trainer.base_trainer import BaseTrainer
from src.utils.audio_utils import MelSpectrogram, MelSpectrogramConfig
from src.logger.utils import plot_spectrogram


class Trainer(BaseTrainer):
    """
    Trainer class. Defines the logic of batch logging and processing.
    """
    def __init__(self, 
                 model, 
                 criterion, 
                 metrics, 
                 optimizer, 
                 lr_scheduler, 
                 config, 
                 device, 
                 dataloaders, 
                 logger, 
                 writer, 
                 epoch_len=None, 
                 skip_oom=True, 
                 batch_transforms=None):
        super().__init__(model, criterion, metrics, optimizer, lr_scheduler, config, device, 
                         dataloaders, logger, writer, epoch_len, skip_oom, batch_transforms)
        mel_spectrogram_config = MelSpectrogramConfig()
        self.mel_spectrogram_transformer = MelSpectrogram(mel_spectrogram_config, device=device)
        self.first_batch_generator = False
        self.first_batch_discriminator = False

        if config.trainer.get("epoch_len", False):
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True 

    def _evaluation_epoch(self, epoch, part, dataloader):
        self.first_batch_generator = True
        self.first_batch_discriminator = True
        return super()._evaluation_epoch(epoch, part, dataloader)

    def process_batch(self, batch, metrics: MetricTracker):
        """
        Run batch through the model, compute metrics, compute loss,
        and do training step (during training stage).

        The function expects that criterion aggregates all losses
        (if there are many) into a single one defined in the 'loss' key.

        Args:
            batch (dict): dict-based batch containing the data from
                the dataloader.
            metrics (MetricTracker): MetricTracker object that computes
                and aggregates the metrics. The metrics depend on the type of
                the partition (train or inference).
        Returns:
            batch (dict): dict-based batch containing the data from
                the dataloader (possibly transformed via batch transform),
                model outputs, and losses.
        """
        batch = self.move_batch_to_device(batch)
        batch = self.transform_batch(batch)  # transform batch on device -- faster

        metric_funcs = self.metrics["inference"]
        if self.is_train:
            metric_funcs = self.metrics["train"]
            self.model.mpd.train()
            self.model.msd.train()

        generated_audios = self.model.generator(batch['mel_spec'])
        batch.update({'generated_audios': generated_audios})
        generated_mpd, _ = self.model.mpd(generated_audios.detach())
        gt_mpd, _ = self.model.mpd(batch['audio'])
        generated_msd, _ = self.model.msd(generated_audios.detach())
        gt_msd, _ = self.model.msd(batch['audio'])
        losses_discriminator = self.criterion.discriminator_loss(gt_mpd, gt_msd, 
                                                                 generated_mpd, generated_msd)
        batch.update(losses_discriminator)
        if self.is_train:
            batch["loss_discriminator"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm(self.model.mpd)
            self._clip_grad_norm(self.model.msd)
            self.optimizer.discriminator_optimizer.step()
        elif self.first_batch_discriminator:
            if self.lr_scheduler is not None:
                self.lr_scheduler.discriminator_scheduler.step()
                self.first_batch = False

        # Generator loss
        if self.is_train:
            self.optimizer.generator_optimizer.zero_grad()
            self.model.mpd.eval()
            self.model.msd.eval()
        
        t = time.time_ns()
        generated_mels = self.mel_spectrogram_transformer(generated_audios)
        batch['spec_generated'] = generated_mels.detach()
        generated_mpd = self.model.mpd(generated_audios)
        generated_msd = self.model.msd(generated_audios)
        gt_mpd = self.model.mpd(batch['audio'])
        gt_msd = self.model.msd(batch['audio'])
        losses_generator = self.criterion.generator_loss(spec_generated=generated_mels,
                                                         probs_generated_mpd=generated_mpd[0],
                                                         probs_generated_msd=generated_msd[0],
                                                         features_generated_mpd=generated_mpd[1],
                                                         features_generated_msd=generated_msd[1],
                                                         features_gt_mpd=gt_mpd[1],
                                                         features_gt_msd=gt_msd[1],
                                                         mel_spec=batch['mel_spec'])
        
        batch.update(losses_generator)

        if self.is_train:
            batch["loss_generator"].backward()  # sum of all losses is always called loss
            self._clip_grad_norm(self.model.generator)
            self.optimizer.generator_optimizer.step()
        elif self.first_batch_generator:
            if self.lr_scheduler is not None:
                self.lr_scheduler.generator_scheduler.step()
                self.first_batch = False

        batch['loss_discriminator'] = batch['loss_discriminator'].detach()
        batch['loss_generator'] = batch['loss_generator'].detach()
        batch['loss'] = (batch['loss_generator'] + batch['loss_discriminator'])

        # update metrics for each loss (in case of multiple losses)
        for loss_name in self.config.writer.loss_names:
            metrics.update(loss_name, batch[loss_name].item())

        for met in metric_funcs:
            metrics.update(met.name, met(**batch))
        return batch

    def _log_batch(self, batch_idx, batch, mode="train"):
        """
        Log data from batch. Calls self.writer.add_* to log data
        to the experiment tracker.

        Args:
            batch_idx (int): index of the current batch.
            batch (dict): dict-based batch after going through
                the 'process_batch' function.
            mode (str): train or inference. Defines which logging
                rules to apply.
        """
        # method to log data from you batch
        # such as audio, text or images, for example

        # logging scheme might be different for different partitions
        if mode == "train":  # the method is called only every self.log_step steps
            self.log_spectrogram(batch['spec_generated'], name='Generated_Spectrogram')
            self.log_spectrogram(batch['mel_spec'], name='Ground_truth_Spectrogram')
            self.log_audio(**batch)
        else:
            # Log Stuff
            # TODO: MOS evaluation via estimator
            self.log_spectrogram(batch['spec_generated'], name='Generated_Spectrogram')
            self.log_spectrogram(batch['mel_spec'], name='Ground_truth_Spectrogram')
            self.log_audio(**batch)

    def log_audio(self, audio_path, examples_to_log=10, **batch):
        gt_audios = batch['audio']
        predicted_audios = batch['generated_audios']
        tuples = list(zip(audio_path, gt_audios, predicted_audios))
        rows = {}
        for path, gt_audio, predicted_audio in tuples[:examples_to_log]:
            rows[Path(path).name] = {
                "estimated": self.writer.add_audio("estimated", predicted_audio.detach().cpu(), MelSpectrogramConfig.sr),
                "target": self.writer.add_audio("target", gt_audio.detach().cpu(), MelSpectrogramConfig.sr),
            }
            self.writer.add_table(
                "predictions", pd.DataFrame.from_dict(rows, orient="index")
            )

    def log_spectrogram(self, spectrogram, name='spectrogram', **batch):
        spectrogram_for_plot = spectrogram[0].detach().cpu()
        image = plot_spectrogram(spectrogram_for_plot)
        self.writer.add_image(name, image)

    def _save_checkpoint(self, epoch, save_best=False, only_best=False):
        """
        Save the checkpoints.

        Args:
            epoch (int): current epoch number.
            save_best (bool): if True, rename the saved checkpoint to 'model_best.pth'.
            only_best (bool): if True and the checkpoint is the best, save it only as
                'model_best.pth'(do not duplicate the checkpoint as
                checkpoint-epochEpochNumber.pth)
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "generator_optimizer": self.optimizer.generator_optimizer.state_dict(),
            "discriminator_optimizer": self.optimizer.discriminator_optimizer.state_dict(),
            "generator_scheduler": self.lr_scheduler.generator_scheduler.state_dict(),
            "discriminator_scheduler": self.lr_scheduler.discriminator_scheduler.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config,
        }
        filename = str(self.checkpoint_dir / f"checkpoint-epoch{epoch}.pth")
        if not (only_best and save_best):
            torch.save(state, filename)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(filename, str(self.checkpoint_dir.parent))
            self.logger.info(f"Saving checkpoint: {filename} ...")
        if save_best:
            best_path = str(self.checkpoint_dir / "model_best.pth")
            torch.save(state, best_path)
            if self.config.writer.log_checkpoints:
                self.writer.add_checkpoint(best_path, str(self.checkpoint_dir.parent))
            self.logger.info("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        """
        Resume from a saved checkpoint (in case of server crash, etc.).
        The function loads state dicts for everything, including model,
        optimizers, etc.

        Notice that the checkpoint should be located in the current experiment
        saved directory (where all checkpoints are saved in '_save_checkpoint').

        Args:
            resume_path (str): Path to the checkpoint to be resumed.
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, self.device)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"]["model"] != self.config["model"]:
            self.logger.warning(
                "Warning: Architecture configuration given in the config file is different from that "
                "of the checkpoint. This may yield an exception when state_dict is loaded."
            )
        self.model.load_state_dict(checkpoint["state_dict"])

        # load optimizer state from checkpoint only when optimizer type is not changed.
        if (
            checkpoint["config"]["generator_optimizer"] != self.config["generator_optimizer"]
            or checkpoint["config"]["discriminator_optimizer"] != self.config["discriminator_optimizer"]
            or checkpoint["config"]["generator_lr_scheduler"] != self.config["generator_lr_scheduler"]
            or checkpoint["config"]["discriminator_lr_scheduler"] != self.config["discriminator_lr_scheduler"]
        ):
            self.logger.warning(
                "Warning: Optimizer or lr_scheduler given in the config file is different "
                "from that of the checkpoint. Optimizer and scheduler parameters "
                "are not resumed."
            )
        else:
            self.optimizer.generator_optimizer.load_state_dict(checkpoint["generator_optimizer"])
            self.optimizer.discriminator_optimizer.load_state_dict(checkpoint["discriminator_optimizer"])
            self.lr_scheduler.generator_scheduler.load_state_dict(checkpoint["generator_scheduler"])
            self.lr_scheduler.discriminator_scheduler.load_state_dict(checkpoint["discriminator_scheduler"])

        self.logger.info(
            f"Checkpoint loaded. Resume training from epoch {self.start_epoch}"
        )

    def _from_pretrained(self, pretrained_path):
        """
        Init model with weights from pretrained pth file.

        Notice that 'pretrained_path' can be any path on the disk. It is not
        necessary to locate it in the experiment saved dir. The function
        initializes only the model.

        Args:
            pretrained_path (str): path to the model state dict.
        """
        pretrained_path = str(pretrained_path)
        if hasattr(self, "logger"):  # to support both trainer and inferencer
            self.logger.info(f"Loading model weights from: {pretrained_path} ...")
        else:
            print(f"Loading model weights from: {pretrained_path} ...")
        checkpoint = torch.load(pretrained_path, self.device)

        if checkpoint.get("state_dict") is not None:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)