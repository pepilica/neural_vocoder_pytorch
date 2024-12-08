from dataclasses import dataclass
import torch
import torchaudio
from torch import nn
from speechbrain.inference.TTS import Tacotron2


from src.utils.audio_utils import MelSpectrogramConfig


class SynthesizeText(nn.Module):
    def __init__(self, config: MelSpectrogramConfig):
        super().__init__()
        self.config = config
        self.tacotron = Tacotron2.from_hparams(
            source="speechbrain/tts-tacotron2-ljspeech", 
            savedir="pretrained_models/tts-tacotron2-ljspeech", 
            overrides={"max_decoder_steps": 10000}
        )
        self.pad_value = config.pad_value

    def forward(self, input_text) -> torch.Tensor:
        mel_output, _, _ = self.tacotron.encode_text(input_text)
        return mel_output.squeeze(0)

