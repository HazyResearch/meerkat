from __future__ import annotations

import abc
import logging
import os
from typing import Sequence, Text

import yaml
from yaml.representer import Representer

from meerkat.columns.list_column import ListColumn
from meerkat.tools.lazy_loader import LazyLoader

spacy = LazyLoader("spacy")
spacy_attrs = LazyLoader("spacy.attrs")
spacy_tokens = LazyLoader("spacy.tokens")

Representer.add_representer(abc.ABCMeta, Representer.represent_name)

logger = logging.getLogger(__name__)


class SpacyColumn(ListColumn):
    def __init__(
        self,
        data: Sequence[spacy_tokens.Doc] = None,
        *args,
        **kwargs,
    ):
        super(SpacyColumn, self).__init__(data=data, *args, **kwargs)

    @classmethod
    def from_docs(cls, data: Sequence[spacy_tokens.Doc], *args, **kwargs):
        return cls(data=data, *args, **kwargs)

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[Text],
        lang: str = "en_core_web_sm",
        *args,
        **kwargs,
    ):
        # Create the pipeline
        nlp = spacy.load(lang)
        return cls(data=[nlp(text) for text in texts], *args, **kwargs)

    @property
    def docs(self):
        return self.data

    @property
    def tokens(self):
        return [list(doc) for doc in self]

    def __getattr__(self, item):
        try:
            return [getattr(doc, item) for doc in self]
        except AttributeError:
            raise AttributeError(f"Attribute {item} not found.")

    @classmethod
    def read(
        cls,
        path: str,
        nlp: spacy.language.Language = None,
        lang: str = None,
        *args,
        **kwargs,
    ) -> SpacyColumn:
        assert (nlp is None) != (lang is None)
        if nlp is None:
            nlp = spacy.load(lang)

        # Load in the data
        metadata = dict(
            yaml.load(open(os.path.join(path, "meta.yaml")), Loader=yaml.FullLoader)
        )
        assert metadata["dtype"] == cls

        # Load the `DocBin` from disk
        docbin = spacy_tokens.DocBin().from_disk(os.path.join(path, "data.spacy"))
        return cls(list(docbin.get_docs(nlp.vocab)))

    def write(self, path: str, **kwargs) -> None:
        # Construct the metadata
        state = self.get_state()
        del state["_data"]
        metadata = {
            "dtype": type(self),
            "len": len(self),
            "state": state,
            **self.metadata,
        }

        # Make directory
        os.makedirs(path, exist_ok=True)

        # Get the paths where metadata and data should be stored
        metadata_path = os.path.join(path, "meta.yaml")
        data_path = os.path.join(path, "data.spacy")

        # Create a `DocBin` to store the docs
        attrs = [name for name in spacy_attrs.NAMES if name != "HEAD"]
        docbin = spacy_tokens.DocBin(attrs=attrs, store_user_data=True, docs=self.docs)

        # Save all the docs
        docbin.to_disk(data_path)

        # Save the metadata as a yaml
        yaml.dump(metadata, open(metadata_path, "w"))
