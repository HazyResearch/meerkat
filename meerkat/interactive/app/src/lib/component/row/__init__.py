import numpy as np
from meerkat.dataframe import DataFrame
from typing import Any, Dict, Optional
from meerkat.interactive.edit import EditTarget
from meerkat.interactive.graph import Store

from ..abstract import Component
from meerkat.interactive.endpoint import Endpoint
from dataclasses import dataclass, field

@dataclass
class Row(Component):

    df: "DataFrame"
    # The primary key column.
    primary_key_column: str
    # The Cell specs
    cell_specs: Dict[str, Dict[str, Any]]
    # The selected key. This should be an element in primary_key_col.
    selected_key: Optional[str] = None
    title: str = ""

    # On change should take in 3 arguments:
    # - key: the primary key (key)
    # - column: the column name (column)
    # - value: the new value (value)
    on_change: Endpoint = None

    # @property
    # def props(self):
    #     return {
    #         "df": self.df.config,  # FIXME
    #         "idx": self.idx.config,
    #         "target": self.target.config,
    #         "cell_specs": self.cell_specs,
    #         "title": self.title,
    #     }
