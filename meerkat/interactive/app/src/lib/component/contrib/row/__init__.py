from typing import Any, Dict, Optional

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import Endpoint

from ...abstract import Component


class Row(Component):

    df: "DataFrame"
    # The primary key column.a
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
