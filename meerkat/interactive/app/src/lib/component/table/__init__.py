from meerkat.dataframe import DataFrame
from meerkat.interactive.edit import EditTarget

from ..abstract import Component


class Table(Component):

    df: DataFrame
    edit_target: EditTarget = None
    per_page: int = 100
    column_widths: list = None
