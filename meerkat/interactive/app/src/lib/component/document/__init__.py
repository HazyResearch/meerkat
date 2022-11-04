from meerkat.interactive.graph import Box, make_box, make_store

from ..abstract import Component


class Document(Component):

    name = "Document"

    def __init__(
        self,
        df: Box,
        doc_column: str,
    ) -> None:
        super().__init__()
        self.df = make_box(df)
        self.doc_column = make_store(doc_column)

    @property
    def props(self):
        props = {
            "df": self.df.config,
            "doc_column": self.doc_column.config,
        }
        return props
