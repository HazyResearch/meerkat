"""Unittests for NumpyColumn."""
from __future__ import annotations

import os
from typing import List, Union

import numpy as np
import pytest
import torch
import torchaudio

from meerkat import AudioColumn
from meerkat.columns.abstract import Column
from meerkat.columns.file_column import FileCell
from meerkat.columns.lambda_column import LambdaCell
from meerkat.columns.pandas_column import ScalarColumn

from .abstract import AbstractColumnTestBed


def simple_transform(audio):
    return 2 * audio


def loader(filepath):
    return torchaudio.load(filepath)[0]


class AudioColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "transform": [True, False],
        "use_base_dir": [True, False],
    }

    marks = pytest.mark.audio_col

    def __init__(
        self,
        tmpdir: str,
        length: int = 16,
        transform: bool = False,
        use_base_dir: bool = False,
        seed: int = 123,
    ):
        self.audio_paths = []
        self.audio_arrays = []
        self.data = []

        transform = simple_transform if transform else None

        self.base_dir = tmpdir if use_base_dir else None

        for i in range(0, length):
            # we want the audio to be variable length to test the collate
            audio = torch.tensor(
                (1 / (i + 1)) * np.ones((1, 16 + i)).astype(np.float32)
            )
            self.audio_arrays.append(audio)
            self.data.append(transform(audio) if transform else audio)
            filename = "{}.wav".format(i)
            torchaudio.save(
                os.path.join(tmpdir, filename),
                torch.tensor(audio),
                sample_rate=16,
            )
            if use_base_dir:
                self.audio_paths.append(filename)
            else:
                self.audio_paths.append(os.path.join(tmpdir, filename))

        self.transform = transform
        self.col = AudioColumn.from_filepaths(
            self.audio_paths,
            transform=transform,
            loader=loader,
            base_dir=self.base_dir,
        )

    def get_data(self, index, materialize: bool = True):
        if materialize:
            if isinstance(index, int):
                return self.data[index]
            index = np.arange(len(self.data))[index]
            return [self.data[idx] for idx in index]

        else:
            if isinstance(index, int):
                return FileCell(
                    data=self.audio_paths[index],
                    loader=self.col.loader,
                    transform=self.col.transform,
                    base_dir=self.base_dir,
                )
            index = np.arange(len(self.data))[index]
            return ScalarColumn([self.audio_paths[idx] for idx in index])

    @staticmethod
    def assert_data_equal(
        data1: Union[Column, List, torch.Tensor],
        data2: Union[Column, List, torch.Tensor],
    ):
        def unpad_and_compare(padded: torch.Tensor, data: List):
            for row_idx in range(padded.shape[0]):
                padded_row = padded[row_idx]
                unpadded_row = padded_row[padded_row != 0]
                assert torch.allclose(unpadded_row, data[row_idx])

        if isinstance(data1, Column) and isinstance(data2, Column):
            assert data1.is_equal(data2)
        elif torch.is_tensor(data1) and torch.is_tensor(data2):
            assert torch.allclose(data1, data2)
        elif torch.is_tensor(data1) and isinstance(data2, List):
            # because the waveforms are of different lengths, collate will put them
            # into a padded tensor, so we use unpad_and_compare to compare to the
            # original unpadded data
            unpad_and_compare(data1, data2)
        elif torch.is_tensor(data2) and isinstance(data1, List):
            unpad_and_compare(data2, data1)
        elif isinstance(data1, LambdaCell):
            assert data1 == data2
        else:
            raise ValueError(
                "Cannot assert data equal between objects type:"
                f" {type(data1), type(data2)}"
            )
