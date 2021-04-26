from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Sequence, Union

from dosma.data_io.format_io import DataReader, ImageDataFormat
from dosma.data_io.format_io_utils import get_reader

from robustnessgym.mosaic.cells.abstract import AbstractCell


class MedicalVolumeCell(AbstractCell):
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
        path: Union[str, Path, Sequence[Union[str, Path]]],
        loader: Callable = None,
        transform: Callable = None,
    ):
        # TODO (arjundd): Convert path(s) to string b/c of DOSMA data loader.
        # Make DOSMA issue to support Path objects.
        if isinstance(path, Path):
            path = str(path)
        elif not isinstance(path, str) and isinstance(path, Sequence):
            path = [str(p) if isinstance(p, str) else p for p in path]

        self.path = path
        self.transform = transform
        self.loader = self.get_default_reader(path) if loader is None else loader
        self.transform = transform

    def get_default_reader(self, path):
        # TODO (arjundd): Make issue in DOSMA asking them to handle these cases.
        if isinstance(self.path, (str, Path, os.PathLike)):
            path = self.path
        else:
            path = self.path[0]
        # TODO (arjundd): Make issue in DOSMA to allow loading with path like objects.
        path = str(path)
        return get_reader(ImageDataFormat.get_image_data_format(path))

    def default_loader(self, *args, **kwargs):
        # TODO (arjundd): If all cells do not need this, remove.
        return self.get_default_reader(self.path)(*args, **kwargs)

    def get(self, *args, **kwargs):
        image = self.loader(self.path)
        if isinstance(image, (list, tuple)) and len(image) == 1:
            image = image[0]
        if self.transform is not None:
            image = self.transform(image)
        return image

    def encode(self):
        is_dosma_reader = isinstance(self.loader, DataReader)
        loader_state = self.loader.state_dict() if is_dosma_reader else self.loader
        loader_type = type(self.loader) if is_dosma_reader else None

        return {
            "path": self.path,
            "loader": {"state": loader_state, "type": loader_type},
            "transform": self.transform,
        }

    @classmethod
    def decode(cls, encoding):
        loader_type = encoding["loader"]["type"]
        loader_state = encoding["loader"]["state"]
        if loader_type is None:
            loader = loader_state
        else:
            loader = loader_type()
            loader.load_state_dict(loader_state)

        return cls(
            encoding["path"],
            loader=loader,
            transform=encoding["transform"],
        )

    def __getitem__(self, index):
        image = self.get()
        return image[index]

    def __str__(self):
        return f"MedicalVolumeCell({self.path})"

    def __repr__(self):
        return f"MedicalVolumeCell({self.path})"
