from pathlib import Path
import os
from tqdm import tqdm
from src.datasets.base_dataset import BaseDataset
from src.transforms.mel_spec import MelSpectrogramConfig, MelSpectrogram


class CustomDirDataset(BaseDataset):
    """
    Dataset of text samples to synthesize audio.
    """
    def __init__(self, transcription_dir=None, audio_dir=None, query=None, mel_spec_device='cpu', *args, **kwargs):
        assert (transcription_dir, audio_dir, query) != (None, None, None), \
        "Either audio, transcription or query directories should be given"
        data = []
        self.text_mode = True

        if query is not None:
            data.append({"id": "my_query", "text": query})
            super().__init__(data, *args, **kwargs)
            return
        
        if transcription_dir is not None:
            data = []
            for path in Path(transcription_dir).iterdir():
                entry = {}
                if path.suffix == '.txt':
                    entry["id"] = str(path.stem)
                    entry['text_path'] = str(path)
                    with path.open() as f:
                        entry["text"] = f.read().strip()
                if len(entry) > 0:
                    data.append(entry)
        else:
            self.text_mode = False
            data = []
            for path in Path(audio_dir).iterdir():
                entry = {}
                if path.suffix in [".mp3", ".wav", ".flac", ".m4a"]:
                    entry["id"] = str(path.stem)
                    entry['audio_path'] = str(path)
                if len(entry) > 0:
                    data.append(entry)

        if len(data) == 0:
            print("WARNING: no transcriptions provided for synthesis.")

        super().__init__(data, *args, **kwargs)
        
    def __getitem__(self, idx):
        entry = self._index[idx]
        instance_data = {
            "id": entry["id"]
        }
        if "audio_path" in entry.keys():
            audio = self.load_audio(entry["audio_path"])
            instance_data['audio_path'] = entry['audio_path']
            instance_data['audio'] = audio
            mel_spec = self.get_spectrogram(audio)
        if 'text' in entry.keys():
            instance_data['text'] = entry['text']
            instance_data['text_path'] = entry.get('text_path', None)
            mel_spec = self.get_spectrogram(entry['text'])
        instance_data['mel_spec'] = mel_spec
        return instance_data
