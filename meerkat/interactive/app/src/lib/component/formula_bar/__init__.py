"""
"""
from meerkat.dataframe import DataFrame
from meerkat.interactive.graph import Store, store_field

from ..abstract import Component


class FormulaBar(Component):

    df: DataFrame
    against: Store[str]
    col: Store[str] = store_field("")
    text: Store[str] = store_field("")
    title: str = ""
