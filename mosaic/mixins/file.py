import os
from pathlib import Path
from typing import Sequence, Union

PathLikeType = Union[str, Path, os.PathLike]
PathLike = (str, Path, os.PathLike)


class FileMixin:
    """Mixin for adding in single filepath."""

    # TODO(karan): this also actually works on dirs, so rename

    def __init__(self, filepath: Union[str, Path], *args, **kwargs):
        super(FileMixin, self).__init__(*args, **kwargs)

        if isinstance(filepath, str):
            filepath = Path(filepath)

        # Assign the path
        self.filepath = filepath

    def __getattr__(self, item):
        try:
            return getattr(self.filepath, item)
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")


class PathsMixin:
    """Mixin for adding in generic paths."""

    def __init__(
        self,
        paths: Union[PathLikeType, Sequence[PathLikeType]],
        *args,
        **kwargs,
    ):
        super(PathsMixin, self).__init__(*args, **kwargs)

        if isinstance(paths, PathLike):
            paths = [Path(paths)]
        elif not isinstance(paths, str) and isinstance(paths, Sequence):
            paths = [Path(p) for p in paths]
        else:
            raise NotImplementedError

        # Assign the path
        # TODO: make this a property
        self.paths = paths

    def __getattr__(self, item):
        try:
            return [getattr(p, item) for p in self.paths]
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")
