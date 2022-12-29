from meerkat.dataframe import DataFrame

from ..abstract import Component


class SchemaTree(Component):

    name = "SchemaTree"

    def __init__(
        self,
        df: DataFrame,
    ) -> None:
        super().__init__()
        self.df = df

    @property
    def props(self):
        return {
            "df": self.df.config,  # FIXME
        }
