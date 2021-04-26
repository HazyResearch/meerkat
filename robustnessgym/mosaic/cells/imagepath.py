from __future__ import annotations

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

    def __getattr__(self, item):
        return super().__getattr__(item)

    def default_loader(self, *args, **kwargs):
        import torchvision.datasets.folder as folder

        return folder.default_loader(*args, **kwargs)

    def get(self, *args, **kwargs):
        image = self.loader(self.filepath)
        if self.transform is not None:
            image = self.transform(image)
        return image

    def get_state(self):
        return {
            "filepath": self.filepath,
            "loader": self.loader,
            "transform": self.transform,
        }

    @classmethod
    def from_state(cls, state):
        return cls(
            state["filepath"],
            loader=state["loader"],
            transform=state["transform"],
        )

    def __str__(self):
        return f"Image({self.name})"

    def __repr__(self):
        return f"Image({self.name})"
