from __future__ import annotations

import abc
import base64
import logging
from io import BytesIO
from typing import Sequence

import cytoolz as tz
from yaml.representer import Representer

import meerkat as mk
from meerkat.columns.abstract import AbstractColumn
from meerkat.mixins.cloneable import CloneableMixin
from meerkat.tools.lazy_loader import LazyLoader

PIL = LazyLoader("PIL")


Representer.add_representer(abc.ABCMeta, Representer.represent_name)


logger = logging.getLogger(__name__)


# Q. how to handle collate and materialize here? Always materialized but only sometimes
# may want to collate (because collate=True should return a batch-style object, while
# collate=False should return a Column style object).


class ListColumn(AbstractColumn):
    def __init__(
        self,
        data: Sequence = None,
        *args,
        **kwargs,
    ):
        if data is not None:
            data = list(data)
        super(ListColumn, self).__init__(data=data, *args, **kwargs)

    @classmethod
    def from_list(cls, data: Sequence):
        return cls(data=data)

    def batch(
        self,
        batch_size: int = 1,
        drop_last_batch: bool = False,
        collate: bool = True,
        *args,
        **kwargs,
    ):
        for i in range(0, len(self), batch_size):
            if drop_last_batch and i + batch_size > len(self):
                continue
            if collate:
                yield self.collate(self[i : i + batch_size])
            else:
                yield self[i : i + batch_size]

    def _repr_cell(self, index) -> object:
        return self[index]

    def _get_formatter(self) -> callable:
        if not mk.config.DisplayOptions.show_images:
            return None

        max_image_width = mk.config.DisplayOptions.max_image_width
        max_image_height = mk.config.DisplayOptions.max_image_height

        def _image_base64(im):
            with BytesIO() as buffer:
                im.save(buffer, "jpeg")
                return base64.b64encode(buffer.getvalue()).decode()

        def _image_formatter(cell):
            im = cell
            if isinstance(im, PIL.Image.Image):
                im.thumbnail((max_image_width, max_image_height))
                return f'<img src="data:image/jpeg;base64,{_image_base64(im)}">'
            else:
                return repr(cell)

        return _image_formatter

    @classmethod
    def concat(cls, columns: Sequence[ListColumn]):
        data = list(tz.concat([c.data for c in columns]))
        if issubclass(cls, CloneableMixin):
            return columns[0]._clone(data=data)
        return cls.from_list(data)

    def is_equal(self, other: AbstractColumn) -> bool:
        return (self.__class__ == other.__class__) and self.data == other.data
