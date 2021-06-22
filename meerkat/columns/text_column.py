from __future__ import annotations

from typing import Sequence

from mosaic.columns.list_column import ListColumn


class TextOutputColumn(ListColumn):
    def __init__(self, data: Sequence = None, *args, **kwargs):

        super(TextOutputColumn, self).__init__(data=data, *args, **kwargs)
