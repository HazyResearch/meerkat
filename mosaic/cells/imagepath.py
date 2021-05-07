from __future__ import annotations

from collections.abc import Collection

from robustnessgym.mosaic.cells.abstract import AbstractCell
from robustnessgym.mosaic.mixins.file import FileMixin


class ImagePath(FileMixin, AbstractCell):
    """This class acts as an interface to allow the user to manipulate the
    images without actually loading them into memory."""

    def __init__(
        self,
        filepath: str,
        loader: callable = None,
        transform: callable = None,
    ):
        super(ImagePath, self).__init__(filepath=filepath)
        self.transform = transform
        self.loader = self.default_loader if loader is None else loader

    def __str__(self):
        return f"Image({self.name})"

    def __repr__(self):
        return f"Image({self.name})"

    @classmethod
    def default_loader(cls, *args, **kwargs):
        import torchvision.datasets.folder as folder

        return folder.default_loader(*args, **kwargs)

    def get(self, *args, **kwargs):
        image = self.loader(self.filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image

    @classmethod
    def _state_keys(cls) -> Collection:
        return {"filepath", "transform", "loader"}
