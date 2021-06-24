from __future__ import annotations

from meerkat.cells.abstract import AbstractCell
from meerkat.tools.lazy_loader import LazyLoader

spacy = LazyLoader("spacy")
spacy_attrs = LazyLoader("spacy.attrs")
spacy_tokens = LazyLoader("spacy.tokens")


class SpacyCell(AbstractCell):
    def __init__(
        self,
        doc: spacy_tokens.Doc,
        *args,
        **kwargs,
    ):
        super(SpacyCell, self).__init__(*args, **kwargs)

        # Put some data into the cell
        self.doc = doc

    def default_loader(self, *args, **kwargs):
        return self

    def get(self, *args, **kwargs):
        return self.doc

    def get_state(self):
        attrs = [name for name in spacy_attrs.NAMES if name != "HEAD"]
        arr = self.get().to_array(attrs)
        return {
            "arr": arr.flatten(),
            "shape": list(arr.shape),
            "words": [t.text for t in self.get()],
        }

    @classmethod
    def from_state(cls, state, nlp: spacy.language.Language):
        doc = spacy_tokens.Doc(nlp.vocab, words=state["words"])
        attrs = [name for name in spacy_attrs.NAMES if name != "HEAD"]
        return cls(doc.from_array(attrs, state["arr"].reshape(state["shape"])))

    def __getitem__(self, index):
        return self.get()[index]

    def __getattr__(self, item):
        try:
            return getattr(self.get(), item)
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")

    def __repr__(self):
        return f"{self.__class__.__name__}({self.get().__repr__()})"


class LazySpacyCell(AbstractCell):
    def __init__(
        self,
        text: str,
        nlp: spacy.language.Language,
        *args,
        **kwargs,
    ):
        super(LazySpacyCell, self).__init__(*args, **kwargs)

        # Put some data into the cell
        self.text = text
        self.nlp = nlp

    def default_loader(self, *args, **kwargs):
        return self

    def get(self, *args, **kwargs):
        return self.nlp(self.text)

    def get_state(self):
        return {
            "text": self.text,
        }

    @classmethod
    def from_state(cls, state, nlp: spacy.language.Language):
        return cls(text=state["text"], nlp=nlp)

    def __getitem__(self, index):
        return self.get()[index]

    def __getattr__(self, item):
        try:
            return getattr(self.get(), item)
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")

    def __repr__(self):
        snippet = f"{self.text[:15]}..." if len(self.text) > 20 else self.text
        return f"{self.__class__.__name__}({snippet})"
