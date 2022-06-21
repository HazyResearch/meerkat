"""Unittests for NumpyColumn."""
from __future__ import annotations

import os
from typing import List, Union

import numpy as np
import pandas as pd
import pytest
import torch
import torchaudio

import meerkat
from meerkat import AudioColumn
from meerkat.columns.abstract import AbstractColumn
from meerkat.columns.file_column import FileCell
from meerkat.columns.lambda_column import LambdaCell
from meerkat.columns.pandas_column import PandasSeriesColumn

from .abstract import AbstractColumnTestBed, TestAbstractColumn


def simple_transform(audio):
    return 2 * audio


class AudioColumnTestBed(AbstractColumnTestBed):

    DEFAULT_CONFIG = {
        "transform": [True, False],
        "use_base_dir": [True, False],
    }

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
            return PandasSeriesColumn([self.audio_paths[idx] for idx in index])

    @staticmethod
    def assert_data_equal(
        data1: Union[AbstractColumn, List, torch.Tensor],
        data2: Union[AbstractColumn, List, torch.Tensor],
    ):
        def unpad_and_compare(padded: torch.Tensor, data: List):
            for row_idx in range(padded.shape[0]):
                padded_row = padded[row_idx]
                unpadded_row = padded_row[padded_row != 0]
                assert torch.allclose(unpadded_row, data[row_idx])

        if isinstance(data1, AbstractColumn) and isinstance(data2, AbstractColumn):
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


@pytest.fixture
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


class TestAudioColumn(TestAbstractColumn):

    __test__ = True
    testbed_class: type = AudioColumnTestBed
    column_class: type = AudioColumn

    def _get_data_to_set(self, testbed, data_index):
        return np.zeros_like(testbed.get_data(data_index))

    @AudioColumnTestBed.parametrize(single=True, params={"index_type": [np.ndarray]})
    def test_set_item(self, testbed, index_type: type):
        with pytest.raises(ValueError, match="Cannot setitem on a `LambdaColumn`."):
            testbed.col[0] = 0

    @AudioColumnTestBed.parametrize(params={"index_type": [np.array]})
    def test_getitem(self, testbed, index_type: type):
        return super().test_getitem(testbed, index_type=index_type)

    @AudioColumnTestBed.parametrize(params={"n": [1, 2, 3]})
    def test_concat(self, testbed: AbstractColumnTestBed, n: int):
        return super().test_concat(testbed, n=n)

    @AudioColumnTestBed.parametrize()
    def test_copy(self, testbed: AbstractColumnTestBed):
        return super().test_copy(testbed)

    @AudioColumnTestBed.parametrize()
    def test_io(self, tmp_path, testbed):
        # uses the tmp_path fixture which will provide a
        # temporary directory unique to the test invocation,
        # important for dataloader
        col, _ = testbed.col, testbed.data

        path = os.path.join(tmp_path, "test")
        col.write(path)

        new_col = self.column_class.read(path)

        assert isinstance(new_col, self.column_class)
        # can't check if the functions are the same since they point to different
        # methods
        assert col.data.is_equal(new_col.data)

    @AudioColumnTestBed.parametrize()
    def test_pickle(self, testbed):
        super().test_pickle(testbed)

    @AudioColumnTestBed.parametrize(params={"max_rows": [6, 16, 20]})
    def test_repr_pandas(self, testbed, max_rows):
        meerkat.config.display.max_rows = max_rows
        series, _ = testbed.col._repr_pandas_()
        assert isinstance(series, pd.Series)
        assert len(series) == min(len(series), max_rows + 1)

    # we are skipping the map and filter tests, because AudioColumn is a very simple
    # subclass of `FileColumn` – see `ImageColumn` for tests of filecolumn subclasses
    def test_map_return_multiple(self):
        pass

    def test_map_return_single(self):
        pass

    def test_map_return_single_w_kwarg(self):
        pass

    def test_filter_1(self):
        pass
