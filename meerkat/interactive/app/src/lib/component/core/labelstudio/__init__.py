from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component


class LabelStudio(Component):
    df: DataFrame

    def __init__(self, df: DataFrame):
        super().__init__(
            df=df,
        )
