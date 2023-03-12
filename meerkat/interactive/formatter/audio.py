import base64
import os
from io import BytesIO
from typing import Any, Dict, Union

from scipy.io.wavfile import write

from meerkat.cells.audio import Audio as AudioCell
from meerkat.columns.deferred.base import DeferredCell
from meerkat.interactive.formatter.icon import IconFormatter

from ..app.src.lib.component.core.audio import Audio
from .base import BaseFormatter, FormatterGroup


class AudioFormatter(BaseFormatter):
    component_class = Audio
    data_prop: str = "data"

    def __init__(self, downsampling_factor: int = 1, classes: str = ""):
        """
        Args:
            sampling_rate_factor: The factor by which to multiply the sampling rate.
            classes: The CSS classes to apply to the audio.
        """
        self.downsampling_factor = downsampling_factor
        self.classes = classes

    def encode(self, cell: Union[AudioCell, bytes]) -> str:
        """Encodes audio as a base64 string.

        Args:
            cell: The image to encode.
            skip_copy: If True, the image may be modified in place.
                Set to ``True`` if the image is already a copy
                or is loaded dynamically (e.g. DeferredColumn).
                This may save time for large images.
        """
        if isinstance(cell, DeferredCell):
            cell = cell()

        # If the audio is already encoded, return it.
        if isinstance(cell, bytes):
            return "data:audio/mpeg;base64,{im_base_64}".format(
                im_base_64=base64.b64encode(cell).decode()
            )

        arr = cell.data
        sampling_rate = cell.sampling_rate

        import torch

        if isinstance(arr, torch.Tensor):
            arr = arr.cpu().numpy()

        arr = arr[:: self.downsampling_factor]
        sampling_rate = sampling_rate // self.downsampling_factor

        with BytesIO() as buffer:
            write(buffer, sampling_rate, arr)
            return "data:audio/mpeg;base64,{im_base_64}".format(
                im_base_64=base64.b64encode(buffer.getvalue()).decode()
            )

    @property
    def props(self) -> Dict[str, Any]:
        return {"classes": self.classes}

    def html(self, cell: AudioCell) -> str:
        encoded = self.encode(cell)
        return (
            "<audio controls={true} autoplay={false}>"
            + f"<source src={encoded} /></audio>"
        )

    def _get_state(self) -> Dict[str, Any]:
        return {"classes": self.classes}

    def _set_state(self, state: Dict[str, Any]):
        self.classes = state["classes"]


class AudioFormatterGroup(FormatterGroup):
    formatter_class: type = AudioFormatter

    def __init__(self, classes: str = ""):
        super().__init__(
            icon=IconFormatter(name="Soundwave"),
            base=self.formatter_class(),
        )


class DeferredAudioFormatter(AudioFormatter):
    component_class: type = Audio
    data_prop: str = "data"

    def encode(self, audio: DeferredCell) -> str:
        if hasattr(audio, "absolute_path"):
            absolute_path = audio.absolute_path
            if isinstance(absolute_path, os.PathLike):
                absolute_path = str(absolute_path)
            if isinstance(absolute_path, str) and absolute_path.startswith("http"):
                return audio.absolute_path

        audio = audio()
        return super().encode(audio)


class DeferredAudioFormatterGroup(AudioFormatterGroup):
    formatter_class: type = DeferredAudioFormatter
