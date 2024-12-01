from tqdm.auto import tqdm
import torchaudio
from pathlib import Path
import random
from src.utils.audio_utils import MelSpectrogramConfig, MelSpectrogram


class LJSpeechDataset:
    def __init__(self, data_dir, audio_max_len=None, limit=None, random=False, device='cpu', **kwargs):
        data_dir_path = Path(data_dir)
        self.paths = []
        for wav_i in data_dir_path.iterdir():
            self.paths.append(wav_i)
        if limit is not None:
            self.paths = self.paths[:limit]
        self.random = random
        mel_spectrogram_config = MelSpectrogramConfig()
        self.mel_spectrogram_transformer = MelSpectrogram(mel_spectrogram_config, device)
        self.wav_max_len = audio_max_len

    def __getitem__(self, index):
        audio_i = torchaudio.load(str(self.paths[index]))[0]
        if self.wav_max_len is not None:
            if self.random:
                start_pos = random.randint(0, audio_i.shape[-1] - self.wav_max_len)
            else:
                start_pos = 0
            audio_i = audio_i[:, start_pos: start_pos + self.wav_max_len]
        mel_spec_i = self.mel_spectrogram_transformer(audio_i.detach())
        if len(audio_i.shape) > 1:
            audio_i = audio_i.mean(dim=0)
            mel_spec_i = mel_spec_i.mean(dim=0)
        return {
            "audio": audio_i,
            "mel_spec": mel_spec_i,
            'audio_path': self.paths[index]
        }

    def __len__(self):
        return len(self.paths)
    