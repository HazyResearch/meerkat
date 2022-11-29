import numpy as np
from meerkat.dataframe import DataFrame
from typing import Any, Dict, Optional
from meerkat.interactive.edit import EditTarget
from meerkat.interactive.graph import Store

from ..abstract import Component
from meerkat.interactive.endpoint import Endpoint
from dataclasses import dataclass, field


class Row(Component):

    df: "DataFrame"
    # The primary key column.a
    primary_key_column: Store[str]
    # The Cell specs
    cell_specs: Store[Dict[str, Dict[str, Any]]]
    # The selected key. This should be an element in primary_key_col.
    selected_key: Store[str] = None
    title: str = ""

    # On change should take in 3 arguments:
    # - key: the primary key (key)
    # - column: the column name (column)
    # - value: the new value (value)
    on_change: Endpoint = None

