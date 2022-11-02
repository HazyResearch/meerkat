from meerkat.interactive.graph import Box, make_box, make_store

from ..abstract import Component


class Document(Component):

    name = "Document"

    def __init__(
        self,
        dp: Box,
        doc_column: str,
    ) -> None:
        super().__init__()
        self.dp = make_box(dp)
        self.doc_column = make_store(doc_column)

    @property
    def props(self):
        props = {
            "dp": self.dp.config,
            "doc_column": self.doc_column.config,
        }
        return props
