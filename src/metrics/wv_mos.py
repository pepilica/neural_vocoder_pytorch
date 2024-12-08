import torch
from wvmos import get_wvmos
from src.metrics.base_metric import BaseMetric


class WVMOSMetric(BaseMetric):
    def __init__(self, audio_paths, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.model = get_wvmos(cuda=True)
        self.audio_paths = audio_paths

    def __call__(self, audio_path: list[str], **kwargs):
        mos_score = self.model.calculate_dir(audio_path)
        return mos_score
