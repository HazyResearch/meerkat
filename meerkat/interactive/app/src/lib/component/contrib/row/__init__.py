from typing import Any, Dict, List, Optional

from meerkat.dataframe import DataFrame
from meerkat.interactive.endpoint import EndpointProperty

from ...abstract import Component


class Row(Component):

    df: "DataFrame"
    columns: List[str] 
    stat_columns: List[str] = {}
    rename: Dict[str, str] = {}
    # The selected key. This should be an element in primary_key_col.
    selected_key: Optional[str] = None
    title: str = ""

    # On change should take in 3 arguments:
    # - key: the primary key (key)
    # - column: the column name (column)
    # - value: the new value (value)
    on_change: EndpointProperty = None
