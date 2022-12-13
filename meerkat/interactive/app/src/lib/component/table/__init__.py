from meerkat.dataframe import DataFrame

from ..abstract import Component


class Table(Component):
    df: DataFrame
    per_page: int = 100
