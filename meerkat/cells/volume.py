from __future__ import annotations

from pathlib import Path
from typing import Callable, Dict, Sequence, Union

from meerkat.cells.abstract import AbstractCell
from meerkat.mixins.cloneable import StateClass
from meerkat.mixins.file import PathLikeType, PathsMixin
from meerkat.tools.lazy_loader import LazyLoader

pydicom = LazyLoader("pydicom")
dosma = LazyLoader("dosma")
dosma_core_io_format_io = LazyLoader("dosma.core.io.format_io")

try:
    # Mapping from pydicom types to python types
    _PYDICOM_TO_PYTHON = {
        pydicom.valuerep.DSdecimal: float,
        pydicom.valuerep.DSfloat: float,
        pydicom.valuerep.IS: int,
        pydicom.valuerep.PersonName: str,
        pydicom.multival.MultiValue: list,
        pydicom.sequence.Sequence: list,
    }
except ImportError:
    _PYDICOM_TO_PYTHON = {}


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
        cache_metadata: bool = False,
        *args,
        **kwargs,
    ):
        super(MedicalVolumeCell, self).__init__(paths=paths, *args, **kwargs)
        self._metadata = None
        self.transform: Callable = transform
        self.cache_metadata = cache_metadata
        self.loader = self.default_loader(self.paths) if loader is None else loader

    def __getitem__(self, index):
        image = self.get()
        return image[index]

    def __str__(self):
        return f"{self.__class__.__name__}({self.paths})"

    def __repr__(self):
        return f"{self.__class__.__name__}({self.paths})"

    @classmethod
    def _unroll_path(cls, paths: Sequence[Path]):
        if len(paths) == 1:
            return paths[0]
        return paths

    @classmethod
    def default_loader(cls, paths: Sequence[Path], *args, **kwargs):
        paths = cls._unroll_path(paths)
        return dosma.get_reader(dosma.ImageDataFormat.get_image_data_format(paths))

    def get(self, *args, cache_metadata: bool = None, **kwargs):
        image = self.loader(self._unroll_path(self.paths))
        if cache_metadata is None:
            cache_metadata = self.cache_metadata
        # DOSMA returns a list of MedicalVolumes by default.
        # RG overrides this functinality  - if only one MedicalVolume
        # is returned, unpack that volume from the list.
        if isinstance(image, (list, tuple)) and len(image) == 1:
            image = image[0]
        if self._metadata is None and cache_metadata:
            _img = image[0] if isinstance(image, (list, tuple)) else image
            headers = _img.headers(flatten=True)
            self._metadata = self._prune_metadata(headers[0]) if headers else None
        if self.transform is not None:
            image = self.transform(image)
        return image

    def _prune_metadata(self, metadata):
        """Prunes the metadata to avoid keeping large fields.

        We delete the PixelData field, because that data is a duplicate
        of the image.
        """
        del metadata["PixelData"]
        return metadata

    def get_metadata(
        self,
        ignore_bytes: bool = False,
        readable: bool = False,
        as_raw_type: bool = False,
        force_load: bool = False,
    ) -> Dict:
        if self._metadata is None:
            if force_load:
                _ = self.get(cache_metadata=True)
                metadata = self._metadata
                if not self.cache_metadata:
                    self.clear_metadata()
            else:
                return None
        else:
            metadata = self._metadata

        # Raw data elements need to be decoded.
        metadata = {
            k: metadata[k] if hasattr(v, "is_raw") and v.is_raw else v
            for k, v in metadata.items()
        }

        if ignore_bytes:
            metadata = {
                k: v for k, v in metadata.items() if not isinstance(v.value, bytes)
            }
        if readable:
            metadata = {v.name: v for v in metadata.values()}
        if as_raw_type:
            metadata = {
                k: (
                    _PYDICOM_TO_PYTHON[type(v.value)](v.value)
                    if type(v.value) in _PYDICOM_TO_PYTHON
                    else v.value
                )
                for k, v in metadata.items()
            }

        return metadata

    def clear_metadata(self):
        self._metadata = None

    def get_state(self):
        # Check if the loader is a `DataReader` from `dosma`
        is_dosma_reader = isinstance(self.loader, dosma_core_io_format_io.DataReader)

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
