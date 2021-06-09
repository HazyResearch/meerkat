import os
import shutil
import unittest

import cv2
import numpy as np
import torch
from parameterized import parameterized
from torchvision.transforms import ColorJitter, Compose

from mosaic.cells.video import VideoCell
from mosaic.contrib.video_corruptions.transforms import (
    TemporalCrop,
    TemporalDownsampling,
)

TEMP_DIR = "./tmp"
MOCK_VIDEO_FILE = f"{TEMP_DIR}/video.mp4"
MOCK_VIDEO_LENGTH = 128
MOCK_VIDEO_SIZE = 32
MOCK_VIDEO_FPS = 29.5
DEFAULT_N_CLIPS = 4
DEFAULT_CLIP_LENGTH = 8
DEFAULT_DOWNSAMPLE_RATIO = 2


class TestVideoCell(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert not os.path.isdir(TEMP_DIR), f"Temp. dir {TEMP_DIR} already exists!"
        print("Creating mock video...")
        os.mkdir(TEMP_DIR)
        np.random.seed(42)
        video_arr = np.random.randint(
            0,
            255,
            (MOCK_VIDEO_LENGTH, MOCK_VIDEO_SIZE, MOCK_VIDEO_SIZE, 3),
            dtype=np.uint8,
        )
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            MOCK_VIDEO_FILE, fourcc, MOCK_VIDEO_FPS, (MOCK_VIDEO_SIZE, MOCK_VIDEO_SIZE)
        )
        for i in range(MOCK_VIDEO_LENGTH):
            frame = cv2.cvtColor(video_arr[i], cv2.COLOR_RGB2BGR)
            writer.write(frame)
        writer.release()
        assert os.path.isfile(MOCK_VIDEO_FILE)
        print(f"Mock video file created at {os.path.abspath(MOCK_VIDEO_FILE)}")
        cls.vid_tensor = VideoCell(MOCK_VIDEO_FILE, time_dim=1).get()

    @classmethod
    def tearDownClass(cls):
        print(f"\nRemoving temp. dir recursively: {os.path.abspath(TEMP_DIR)}")
        shutil.rmtree(TEMP_DIR)

    def get_video(self, transform=None):
        cell = VideoCell(MOCK_VIDEO_FILE, time_dim=1, transform=transform)
        vid = cell.get()
        return vid

    def test_basic_construction(self):
        vid = self.get_video()
        self.assertEqual(
            vid.size(), (3, MOCK_VIDEO_LENGTH, MOCK_VIDEO_SIZE, MOCK_VIDEO_SIZE)
        )
        self.assertTrue(torch.equal(self.vid_tensor, vid))

    def test_nonexistent_filepath(self):
        cell = VideoCell("this_file_should_not_exist.mp4", time_dim=1)
        with self.assertRaises(ValueError):
            _ = cell.get()

    @parameterized.expand(
        [
            (0.6,),
            (-4,),
        ]
    )
    def test_invalid_clip_length(self, clip_len):
        with self.assertRaises(ValueError):
            _ = TemporalCrop(DEFAULT_N_CLIPS, clip_len, time_dim=1)

    @parameterized.expand([(2,), (4,), (5,), (16,), (1024,)])
    def test_clip_lengths(self, clip_len):
        crop_fn = TemporalCrop(DEFAULT_N_CLIPS, clip_len, time_dim=1)
        vid = self.get_video(transform=crop_fn)
        self.assertEqual(
            vid.size(), (DEFAULT_N_CLIPS, 3, clip_len, MOCK_VIDEO_SIZE, MOCK_VIDEO_SIZE)
        )
        looped_vid = torch.cat(
            [self.vid_tensor] * (clip_len // MOCK_VIDEO_LENGTH + 2), dim=1
        )
        for i, vid in enumerate(torch.split(vid, 1, dim=0)):
            start = i * MOCK_VIDEO_LENGTH // DEFAULT_N_CLIPS
            target_clip = torch.index_select(
                looped_vid, 1, torch.arange(start, start + clip_len)
            )
            self.assertTrue(torch.equal(vid.squeeze(0), target_clip))

    @parameterized.expand(
        [
            (0.6,),
            (-4,),
        ]
    )
    def test_invalid_n_clips(self, clip_len):
        with self.assertRaises(ValueError):
            _ = TemporalCrop(clip_len, MOCK_VIDEO_LENGTH, time_dim=1)

    @parameterized.expand([(1,), (4,), (32,), (1024,)])
    def test_more_clips_than_frames(self, n_clips):
        crop_fn = TemporalCrop(n_clips, DEFAULT_CLIP_LENGTH, time_dim=1)
        vid = self.get_video(transform=crop_fn)
        looped_vid = torch.cat([self.vid_tensor] * 2, dim=1)
        for i, vid in enumerate(torch.split(vid, 1, dim=0)):
            start = i * MOCK_VIDEO_LENGTH // n_clips
            target_clip = torch.index_select(
                looped_vid, 1, torch.arange(start, start + DEFAULT_CLIP_LENGTH)
            )
            self.assertTrue(torch.equal(vid.squeeze(0), target_clip))

    @parameterized.expand(
        [
            (2,),
            (4,),
            (32,),
            (1024,),
        ]
    )
    def test_transform_downsampling_only(self, ratio):
        downsample_fn = TemporalDownsampling(ratio, time_dim=1)
        vid = self.get_video(transform=downsample_fn)
        indices = torch.arange(0, self.vid_tensor.size(1), ratio)
        target_clip = torch.index_select(self.vid_tensor, 1, indices)
        self.assertTrue(torch.equal(vid, target_clip))

    @parameterized.expand(
        [
            (0.6,),
            (-1,),
            (0,),
        ]
    )
    def test_bad_downsampling_ratios(self, bad_ratio):
        with self.assertRaises(ValueError):
            _ = TemporalDownsampling(bad_ratio, time_dim=1)

    @parameterized.expand(
        [
            (10, 16, 2),
            (8, 8, 8),
            (17, 31, 101),
            (150, 2, 67),
            (25, 1, 1),
        ]
    )
    def test_composed_temporal_transforms(self, n_clips, clip_length, ratio):
        transform_fn = Compose(
            [
                TemporalDownsampling(ratio, time_dim=1),
                TemporalCrop(n_clips, clip_length, time_dim=1),
            ]
        )
        vid = self.get_video(transform=transform_fn)
        indices = torch.arange(0, self.vid_tensor.size(1), ratio)
        downsampled_clip = torch.index_select(
            self.vid_tensor, 1, indices
        )  # contingent on previous test case
        looped_clip = torch.cat(
            [downsampled_clip] * (clip_length // downsampled_clip.size(1) + 2), dim=1
        )
        for i, vid in enumerate(torch.split(vid, 1, dim=0)):
            start = i * downsampled_clip.size(1) // n_clips
            target_clip = torch.index_select(
                looped_clip, 1, torch.arange(start, start + clip_length)
            )
            self.assertTrue(torch.equal(vid.squeeze(0), target_clip))

    def test_composition_with_torchvision(self):
        transform_fn = Compose(
            [
                TemporalDownsampling(DEFAULT_DOWNSAMPLE_RATIO, time_dim=1),
                TemporalCrop(DEFAULT_N_CLIPS, DEFAULT_CLIP_LENGTH, time_dim=1),
                ColorJitter(),
            ]
        )
        _ = self.get_video(transform=transform_fn)
        # no assert -- just make sure that there are no errors in the composition.
        # other test cases cover making sure the boundary behavior behaves as expected

    @parameterized.expand(
        [
            (10, 16, 2),
            (8, 8, 8),
            (17, 31, 101),
            (150, 2, 67),
            (25, 1, 1),
        ]
    )
    def test_anywhere_clip_spacing(self, n_clips, clip_length, ratio):
        # mostly a "compilation" test -- hard to predict the clip start locations
        # even with seeding
        transform_fn = Compose(
            [
                TemporalDownsampling(ratio, time_dim=1),
                TemporalCrop(n_clips, clip_length, time_dim=1, clip_spacing="anywhere"),
            ]
        )
        _ = self.get_video(transform=transform_fn)

    def test_time_dim_0(self):
        # most users will only need to set time_dim to 1 (most common), or CTHW,
        # or 0, which is TCHW. (*, 3, H, W) is needed for composition w/
        # torchvision (i.e. time_dim=1).
        transform_fn = Compose(
            [
                TemporalDownsampling(DEFAULT_DOWNSAMPLE_RATIO, time_dim=0),
                TemporalCrop(DEFAULT_N_CLIPS, DEFAULT_CLIP_LENGTH, time_dim=0),
            ]
        )
        _ = self.get_video(transform=transform_fn)

    def test_shape_without_stack_clips(self):
        transform_fn = Compose(
            [
                TemporalDownsampling(DEFAULT_DOWNSAMPLE_RATIO, time_dim=1),
                TemporalCrop(
                    DEFAULT_N_CLIPS, DEFAULT_CLIP_LENGTH, time_dim=1, stack_clips=False
                ),
            ]
        )
        vid = self.get_video(transform=transform_fn)
        self.assertEqual(
            vid.size(),
            (
                3,
                DEFAULT_N_CLIPS * DEFAULT_CLIP_LENGTH,
                MOCK_VIDEO_SIZE,
                MOCK_VIDEO_SIZE,
            ),
        )
        indices = torch.arange(0, self.vid_tensor.size(1), DEFAULT_DOWNSAMPLE_RATIO)
        downsampled_clip = torch.index_select(
            self.vid_tensor, 1, indices
        )  # contingent on previous test case
        looped_clip = torch.cat(
            [downsampled_clip] * (DEFAULT_CLIP_LENGTH // downsampled_clip.size(1) + 2),
            dim=1,
        )
        for i, vid in enumerate(torch.split(vid, DEFAULT_CLIP_LENGTH, dim=1)):
            start = i * downsampled_clip.size(1) // DEFAULT_N_CLIPS
            target_clip = torch.index_select(
                looped_clip, 1, torch.arange(start, start + DEFAULT_CLIP_LENGTH)
            )
            self.assertTrue(torch.equal(vid, target_clip))
