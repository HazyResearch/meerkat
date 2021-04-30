from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence, Union

from dosma.core.io.format_io import DataReader, ImageDataFormat
from dosma.core.io.format_io_utils import get_reader

from robustnessgym.mosaic.cells.abstract import AbstractCell
from robustnessgym.mosaic.mixins.file import PathLikeType, PathsMixin
from robustnessgym.mosaic.mixins.state import StateClass


class MedicalVolumeCell(PathsMixin, AbstractCell):
    """Interface for loading medical volume data.

    Examples:

        # Specify xray dicoms with default orientation ``("SI", "AP")``:
        >>> cell = MedicalVolumeCell("/path/to/xray.dcm",
        loader=DicomReader(group_by=None, default_ornt=("SI", "AP"))

        # Load multi-echo MRI volumes
        >>> cell = MedicalVolumeCell("/path/to/mri/scan/dir",
        loader=DicomReader(group_by="EchoNumbers"))
    """

    def __init__(
        self,
        paths: Union[PathLikeType, Sequence[PathLikeType]],
        loader: Callable = None,
        transform: Callable = None,
        *args,
        **kwargs,
    ):
        super(MedicalVolumeCell, self).__init__(paths=paths, *args, **kwargs)
        self.transform = transform
        self.loader = self.default_loader(self.paths) if loader is None else loader

    def __getitem__(self, index):
        image = self.get()
        return image[index]

    def __str__(self):
        return f"{self.__class__.__name__}({self.name})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name})"

    @classmethod
    def _unroll_path(cls, paths: Sequence[Path]):
        if len(paths) == 1 and os.path.isdir(paths[0]):
            return paths[0]
        return paths

    @classmethod
    def default_loader(cls, paths: Sequence[Path], *args, **kwargs):
        paths = cls._unroll_path(paths)
        return get_reader(ImageDataFormat.get_image_data_format(paths))

    def get(self, *args, **kwargs):
        image = self.loader(self._unroll_path(self.paths))
        # DOSMA returns a list of MedicalVolumes by default.
        # RG overrides this functinality  - if only one MedicalVolume
        # is returned, unpack that volume from the list.
        if isinstance(image, (list, tuple)) and len(image) == 1:
            image = image[0]
        if self._metadata is None:
            _img = image[0] if isinstance(image, (list, tuple)) else image
            headers = _img.headers(flatten=True)
            self._metadata = dict(headers[0]) if headers else {}
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_state(self):
        # Check if the loader is a `DataReader` from `dosma`
        is_dosma_reader = isinstance(self.loader, DataReader)

        loader = StateClass(
            klass=type(self.loader) if is_dosma_reader else None,
            state=self.loader.state_dict() if is_dosma_reader else self.loader,
        )

        return {
            "paths": self.paths,
            "loader": loader,
            "transform": self.transform,
        }

    @classmethod
    def from_state(cls, state, *args, **kwargs):
        # Unpack the state
        loader_state = state["loader"]

        if loader_state.klass is None:
            loader = loader_state.state
        else:
            loader = loader_state.klass()
            loader.load_state_dict(loader_state.state)

        return cls(
            state["paths"],
            loader=loader,
            transform=state["transform"],
        )
