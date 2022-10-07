from typing import Callable

import librosa

from ..tools.lazy_loader import LazyLoader
from meerkat.interactive.formatter import AudioFormatter
from .file_column import FileColumn


class AudioColumn(FileColumn):
    """A lambda column where each cell represents an audio file on disk. The
    underlying data is a `PandasSeriesColumn` of strings, where each string is
    the path to an image. The column materializes the images into memory when
    indexed. If the column is lazy indexed with the ``lz`` indexer, the images
    are not materialized and an ``FileCell`` or an ``AudioColumn`` is returned
    instead.

    Args:
        data (Sequence[str]): A list of filepaths to images.
        transform (callable): A function that transforms the image (e.g.
            ``torchvision.transforms.functional.center_crop``).

            .. warning::
                In order for the column to be serializable, the transform function must
                be pickleable.


        loader (callable): A callable with signature ``def loader(filepath: str) ->
            PIL.Image:``. Defaults to ``torchvision.datasets.folder.default_loader``.

            .. warning::
                In order for the column to be serializable with ``write()``, the loader
                function must be pickleable.

        base_dir (str): A base directory that the paths in ``data`` are relative to. If
            ``None``, the paths are assumed to be absolute.
    """

    @staticmethod
    def _get_default_formatter() -> Callable:
        return AudioFormatter()

    @classmethod
    def default_loader(cls, *args, **kwargs):
        data, sr = librosa.load(*args, **kwargs)
        return {'data': data, 'sample_rate': sr}
