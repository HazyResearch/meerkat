from __future__ import annotations

from typing import Hashable, Sequence, Tuple

from .column_spec import ColumnSpec


class AbstractBlock:
    def __init__(self, *args, **kwargs):
        super(AbstractBlock, self).__init__(*args, **kwargs)

    @property
    def signature(self) -> Hashable:
        raise NotImplementedError

    @classmethod
    def from_data(
        cls, data: object, spec: ColumnSpec
    ) -> Tuple[AbstractBlock, ColumnSpec]:

        return

    @staticmethod
    def consolidate(blocks: Sequence[Tuple[AbstractBlock, ColumnSpec]]):
        pass

    def _get():
        pass
