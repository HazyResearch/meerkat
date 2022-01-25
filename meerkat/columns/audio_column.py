import torch
import torchaudio
from IPython.display import Audio

import meerkat as mk

from .image_column import ImageColumn


class AudioColumn(ImageColumn):
    @classmethod
    def default_loader(cls, *args, **kwargs):
        return torchaudio.load(*args, **kwargs)[0]

    def _repr_cell(self, idx):
        return self.lz[idx]

    def _get_formatter(self) -> callable:
        if not mk.config.DisplayOptions.show_audio:
            return None

        def _audio_formatter(cell):
            return Audio(filename=cell.data)._repr_html_()

        return _audio_formatter

    def collate(self, batch):
        tensors = [b.t() for b in batch]
        tensors = torch.nn.utils.rnn.pad_sequence(tensors, batch_first=True)
        tensors = tensors.transpose(1, -1)
        return tensors
