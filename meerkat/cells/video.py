import os
from typing import Callable, Collection, Optional

import torch

from meerkat import AbstractCell
from meerkat.contrib.video_corruptions.utils import stderr_suppress
from meerkat.tools.lazy_loader import LazyLoader

cv2 = LazyLoader("cv2")
F = LazyLoader("torchvision.transforms.functional")


class VideoCell(AbstractCell):
    """Interface for loading video data.

    Examples:

    # Load a video from "/path/to/video.mp4", where the time dimension has index one
    >>> cell = VideoCell("/path/to/video.mp4", time_dim=1)
    """

    def __init__(
        self,
        filepath: str,
        transform: Optional[Callable] = None,
        time_dim: Optional[int] = 1,
    ):
        super().__init__()
        self.filepath = filepath
        self.time_dim = time_dim
        self.transform = transform

    def _read_all_frames(self):
        if not os.path.isfile(self.filepath):
            raise ValueError(f"{self.filepath} is not a valid file!")
        cap = cv2.VideoCapture(self.filepath)
        frames = []
        ret = True
        while ret:
            ret, img = cap.read()
            try:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            except cv2.error:
                break
            img = F.to_tensor(img)
            frames.append(img)
        cap.release()
        frames = torch.stack(frames, dim=self.time_dim)
        return frames

    def get(self):
        with stderr_suppress():
            frames = self._read_all_frames()  # TODO: support different decoders
        if self.transform is not None:
            frames = self.transform(frames)
        return frames

    @classmethod
    def _state_keys(cls) -> Collection:
        return {"filepath", "time_dim", "transform"}
