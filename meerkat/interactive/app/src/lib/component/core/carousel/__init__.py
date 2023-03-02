from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component


class Carousel(Component):
    df: DataFrame
    main_column: str

    def __init__(self, df: DataFrame, *, main_column: str):
        super().__init__(df=df, main_column=main_column)
