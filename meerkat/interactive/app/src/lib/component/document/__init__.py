from meerkat.interactive.edit import EditTarget
from meerkat.interactive.graph import Box, make_box, make_store

from ..abstract import Component


class Document(Component):

    name = "Document"

    def __init__(
        self,
        df: Box,
        text_column: str,
        paragraph_column: str = None,
        label_column: str = None,
        edit_target: EditTarget = None,
    ) -> None:
        super().__init__()
        self.df = make_box(df)
        self.text_column = make_store(text_column)
        self.paragraph_column = make_store(paragraph_column)
        self.label_column = make_store(label_column)
        self.edit_target = edit_target

    @property
    def props(self):
        props = {
            "df": self.df.config,
            "text_column": self.text_column.config,
            "paragraph_column": self.paragraph_column.config,
            "label_column": self.label_column.config,
        }
        if self.edit_target is not None:
            props["edit_target"] = self.edit_target.config
        return props
