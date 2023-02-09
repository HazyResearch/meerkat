import random
from typing import TYPE_CHECKING, Optional, Tuple

from meerkat.tools.lazy_loader import LazyLoader

torch = LazyLoader("torch")

if TYPE_CHECKING:
    import torch


class TemporalDownsampling(object):
    """Video transformation for performing temporal downsampling (i.e. reading
    in every Nth frame only). This can be used in tandem with VideoCell by
    passing it into the `transform` keyword in the constructor. Can be used
    with Compose in torchvision.

    When using with TemporalCrop, it is highly recommended to put TemporalDownsampling
    first, with TemporalCrop second.

    Arguments:
        downsample_factor (int): the factor by which the input video should be
            downsampled. Must be a strictly positive integer.
        time_dim (int): the time dimension of the input video.

    Examples:

        # Create a VideoCell from "/path/to/video.mp4" with time in dimension one,
        showing every other frame
        >>> cell = VideoCell("path/to/video.mp4",
            time_dim=1,
            transform=TemporalDownsampling(2, time_dim=1)
            )

    Note that time_dim in the TemporalDownsampling call must match the the time_dim
    in the VideoCell constructor!
    """

    def __init__(self, downsample_factor: int, time_dim: Optional[int] = 1):
        self.downsample_factor = downsample_factor
        self.time_dim = time_dim
        if self.downsample_factor != int(self.downsample_factor):
            raise ValueError("Fractional downsampling not supported.")
        if self.downsample_factor < 1:
            raise ValueError(
                "Downsampling must be by a factor of 1 (no upsampling) or greater."
            )

    def __call__(self, video: "torch.Tensor") -> "torch.Tensor":
        video_length = video.size(self.time_dim)
        downsampled_indices = torch.arange(
            0, video_length, self.downsample_factor
        ).long()
        frames = torch.index_select(video, self.time_dim, downsampled_indices)
        return frames


class TemporalCrop(object):
    """Video transformation for performing "temporal cropping:" the sampling of
    a pre-defined number of clips, each with pre-defined length, from a full
    video. Can be used with Compose in torchvision.

    When used with TemporalDownsampling, it is highly recommended to put
    TemporalCrop after TemporalDownsampling. Since TemporalCrop can change the
    number of dimensions in the output tensor, due to clip selection, it is in
    fact recommended to put this transform at the end of a video transformation
    pipeline.

    Arguments:
        n_clips (int): the number of clips that should be sampled.
        clip_length (int): the length of each clip (in the number of frames)
        time_dim (int): the index of the time dimension of the video
        clip_spacing (Optional; default "equal"): how to choose starting locations
            for sampling clips. Keyword "equal" means that clip starting locations
            are sampled from each 1/n_clips segment of the video. The other option,
            "anywhere", places no restrictions on where clip starting locations
            can be sampled.
        padding_mode: (Optional; default "loop"): behavior if a requested clip
            length would result a clip exceeding the end of the video. Keyword
            "loop" results in a wrap-around to the start of the video. The other
            option, "freeze", repeats the final frame until the requested clip
            length is achieved.
        sample_starting_location: (Optional; default False): whether to sample a
            starting location (usually used for training) for a clip. Can be used
            in tandem with "equal" during training to sample clips with random
            starting locations distributed across time. Redundant if `clip_spacing`
            is "anywhere".
        stack_clips: (Optional; default True): whether to stack clips in a new
            dimension (used in 3D action recognition backbones), or stack clips by
            concatenating across the time dimension (used in 2D action recognition
            backbones). Output shape if True is (n_clips, *video_shape). If False,
            the output shape has the same number of dimensions as the original
            video, but the time dimension is extended by a factor of n_clips.


    Examples:

        # Create a VideoCell from "/path/to/video.mp4" with time in dimension one,
        sampling 10 clips each of length 16, sampling clips equally across the video
        >>> cell = VideoCell("/path/to/video.mp4",
            time_dim=1,
            transform=TemporalCrop(10, 16, time_dim=1)
            )
        # output shape: (10, n_channels, 16, H, W)

        # Create a VideoCell from "/path/to/video.mp4" with time in dimension one,
        sampling 8 clips each of length 8, sampling clips from arbitrary video
        locations and freezing the last frame if a clip exceeds the video length
        >>> cell = VideoCell("/path/to/video.mp4",
            time_dim=1,
            transform=TemporalCrop(8, 8, time_dim=1, clip_spacing="anywhere",
            padding_mode="freeze")
            )
        # output shape: (8, n_channels, 8, H, W)

        # Create a VideoCell from "/path/to/video.mp4" with time in dimension one,
        sampling one frame from each third of the video, concatenating the frames
        in the time dimension
        >>> cell = VideoCell("/path/to/video.mp4",
            time_dim=1,
            transform=TemporalCrop(1, 3, time_dim=1, clip_spacing="equal",
                sample_starting_location=True, stack_clips=False)
            )
        # output shape: (n_channels, 3, H, W)

    Note that time_dim in the TemporalDownsampling call must match the the time_dim
    in the VideoCell constructor!
    """

    def __init__(
        self,
        n_clips: int,
        clip_length: int,
        time_dim: Optional[int] = 1,
        clip_spacing: Optional[str] = "equal",
        padding_mode: Optional[str] = "loop",
        sample_starting_location: Optional[bool] = False,
        stack_clips: Optional[bool] = True,
    ):
        self.n_clips = n_clips
        self.clip_length = clip_length
        self.time_dim = time_dim
        self.clip_spacing = clip_spacing
        self.padding_mode = padding_mode
        self.stack_clips = stack_clips
        self.sample_starting_location = sample_starting_location
        if clip_length != int(clip_length):
            raise ValueError("Clip length (# of frames per clip) must be an integer")
        if clip_length <= 0:
            raise ValueError(
                "Clip length (# of frames per clip) must be a positive integer."
            )
        if n_clips != int(n_clips):
            raise ValueError("Number of clips is not an integer")
        if n_clips <= 0:
            raise ValueError("Number of clips must be a positive integer")
        assert clip_spacing in ["equal", "anywhere"]
        assert padding_mode in ["loop", "freeze"]

    def _get_sampling_boundaries(
        self, video_length: int, clip_number: int
    ) -> Tuple[int, int]:
        if self.clip_spacing == "equal":
            start = int(clip_number / self.n_clips * video_length)
            end = int((clip_number + 1) / self.n_clips * video_length)
        else:  # self.clip_spacing == "anywhere"
            start = 0
            end = video_length
        return start, end

    def _build_indices(self, start: int, length: int) -> "torch.LongTensor":
        vanilla_indices = torch.arange(start, start + self.clip_length)
        if vanilla_indices.max().item() >= length:
            if self.padding_mode == "loop":
                vanilla_indices %= length
            else:  # self.padding_mode == "freeze":
                vanilla_indices[vanilla_indices >= length] = length - 1
        return vanilla_indices

    def __call__(self, video: "torch.Tensor") -> "torch.Tensor":
        video_length = video.size(self.time_dim)
        clips = []
        for clip_number in range(self.n_clips):
            start, end = self._get_sampling_boundaries(video_length, clip_number)
            if self.sample_starting_location:
                first_frame = random.randint(start, end)
            else:
                first_frame = start
            indices = self._build_indices(first_frame, video_length)
            clip = torch.index_select(video, self.time_dim, indices)
            clips.append(clip)
        if self.stack_clips:  # new dim for clips (n_clips, n_channels, duration, h, w)
            all_clips = torch.stack(clips, dim=0)
        else:  # concat clips in time dimension (n_channels, n_clips * duration, h, w)
            all_clips = torch.cat(clips, dim=self.time_dim)

        return all_clips
