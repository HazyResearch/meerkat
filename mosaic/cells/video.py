import os
import random
from typing import Callable, Collection, Optional, Tuple

import cv2
import torch
import torchvision.transforms.functional as F

from mosaic import AbstractCell

SAMPLING_MODE_ERR_STR = "Sampling mode {} is not implemented; must be 'semgented' (to sample clips from segments of uniform size) or 'anywhere' (to sample clips from anywhere in the video)."


class stderr_suppress(object):
    '''
    A context manager for doing a "deep suppression" of stdout and stderr in
    Python, i.e. will suppress all print, even if the print originates in a
    compiled C/Fortran sub-function.

    This will not suppress raised exceptions, since exceptions are printed
    to stderr just before a script exits, and after the context manager has
    exited (at least, I think that is why it lets exceptions through). This is 
    necessary when reading in a corrupted video, or else stderr will emit 10000s
    of errors via ffmpeg.
    '''
    def __init__(self):
        # Open a pair of null files
        self.null_fd = os.open(os.devnull, os.O_RDWR)
        # Save stderr (2) file descriptor.
        self.save_fd = os.dup(2)

    def __enter__(self):
        # Assign the null pointers to stdout and stderr.
        os.dup2(self.null_fd, 2)

    def __exit__(self, *_):
        # Re-assign the real stdout/stderr back to (1) and (2)
        os.dup2(self.save_fd, 2)
        # Close all file descriptors
        os.close(self.null_fd)
        os.close(self.save_fd)


class VideoCell(AbstractCell):

    # What information will we eventually  need to materialize the cell? 
    def __init__(self,
            filepath: str,
            clip_length: int,
            n_clips: int,
            transform: Optional[Callable] = None,
            clip_sampling: Optional[str] = "equal_segment",
            downsample_ratio: Optional[int] = 1,
            padding_mode: Optional[str] = "loop",
            random_clip_start: Optional[bool] = True,
            time_dim: Optional[int] = 1,
            stack_clips: Optional[bool] = True):
        super().__init__()
        self.filepath = filepath

        # TODO: abstract into a downsampling transform
        self.downsample_ratio = downsample_ratio

        # TODO: abstract into a clip-ifier transform
        self.clip_length = clip_length
        self.n_clips = n_clips
        self.stack_clips = stack_clips
        self.clip_sampling = clip_sampling
        self.padding_mode = padding_mode # clip-ifier needs to include padding!
        self.random_clip_start = random_clip_start
        self.frame_indices = None

        # pass into both new transforms
        self.time_dim = time_dim

        self.transform = transform

    def read_all_frames(self):
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
            # img = F.resize(img, self.size)
            frames.append(img)
        cap.release()
        frames = torch.stack(frames, dim=self.time_dim)
        return frames


    def get_sampling_boundaries(self, video_length: int, clip_number: int) -> Tuple[int, int]:
        if self.clip_sampling == "equal_segment":
            start = int(clip_number / self.n_clips * video_length)
            end = int((clip_number + 1) / self.n_clips * video_length)
        elif self.clip_sampling == "anywhere":
            start = 0
            end = video_length
        else:
            raise ValueError(SAMPLING_MODE_ERR_STR.format(self.clip_sampling))
        return start, end


    def temporal_padding(self, inp: torch.Tensor, pad_to: int) -> torch.Tensor:
        while inp.size(self.time_dim) < pad_to:
            if self.padding_mode == "loop":  # default: loop video until length == pad_to
                inp = torch.cat([inp, inp], dim=self.time_dim)
                if inp.size(self.time_dim) > pad_to:
                    inp = torch.index_select(inp, self.time_dim, torch.arange(pad_to))
                    # equiv to inp[..., :pad_to, ...] but `pad_to` axis is dynamic
            elif self.padding_mode == "freeze":  # repeat last frame
                indices = torch.clip(torch.arange(pad_to), 0, inp.size(self.time_dim) - 1)
                inp = torch.index_select(inp, self.time_dim, indices)
        return inp

    def build_indices(self, start, length):
        vanilla_indices = torch.arange(start, start + self.clip_length)
        if vanilla_indices.max().item() >= length:
            if self.padding_mode == "loop":
                vanilla_indices %= length
            elif self.padding_mode == "freeze":
                vanilla_indices[vanilla_indices >= length] = length - 1
            else:
                raise ValueError(f"Padding mode must be 'loop' or 'freeze' but got {self.padding_mode}")
        return vanilla_indices

    def temporal_crop(self, frames):
        video_length = frames.size(self.time_dim)
        clips = []
        self.frame_indices = []
        # temporal up/downsampling
        if self.downsample_ratio != 1:
            downsampled_indices = torch.arange(0, video_length, self.downsample_ratio).long()
            frames = torch.index_select(frames, self.time_dim, downsampled_indices)

        for clip_number in range(self.n_clips):
            start, end = self.get_sampling_boundaries(video_length, clip_number)
            if self.random_clip_start:
                first_frame = random.randint(start, end - self.clip_length)
            else:
                first_frame = start
            indices = self.build_indices(first_frame, len(frames))
            self.frames_indices.append(indices.numpy())
            clip = torch.index_select(frames, self.time_dim, indices)
            clips.append(clip)
        if self.stack_clips:  # new dim. for clips (n_clips, channel, duration, height, width)
            return torch.stack(clips, dim=0)
            self.time_dim += 1
        else:  # concatenate clips in time dimension (channel, n_clips * duration, height, width)
            return torch.cat(clips, dim=self.time_dim)

    def get(self):
        with stderr_suppress():
            frames = self.read_all_frames() # TODO: support different decoders
        frames = self.temporal_crop(frames) # TODO: possibly combine with first step to surface tensor
        if self.transform is not None:
            frames = self.transform(frames)
        return frames

    @classmethod
    def _state_keys(cls) -> Collection:
        return {"filepath", "n_clips", "downsample_ratio", "clip_length", "clip_sampling", "padding_mode", "random_clip_start", "time_dim", "stack_clips", "transform"}


