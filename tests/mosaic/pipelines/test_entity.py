"""Unittests for Datasets."""
from itertools import product

import numpy as np
import pytest
import torch

from mosaic import EmbeddingColumn
from mosaic.pipelines.entity import Entity


def _get_entity_data():
    # Build a dataset from a batch
    data = {
        "a": [1, 2, 3],
        "b": [True, False, True],
        "c": ["x", "y", "z"],
        "d": [{"e": 2}, {"e": 3}, {"e": 4}],
        "e": torch.ones(3),
        "f": np.ones(3),
        "g": np.random.rand(3, 5),
    }
    return data


@pytest.mark.parametrize(
    "with_embs, with_index",
    product([True, False], [True, False]),
)
def test_load(with_embs, with_index):
    data = _get_entity_data()
    if with_embs:
        if with_index:
            ent = Entity(data, embedding_columns=["g"], index_column="a")
        else:
            ent = Entity(data, embedding_columns=["g"])
    else:
        if with_index:
            ent = Entity(data, index_column="a")
        else:
            ent = Entity(data)

    if with_embs:
        assert ent.embedding_columns == ["g"]
        assert isinstance(ent["g"], EmbeddingColumn)
    else:
        assert ent.embedding_columns == []

    if with_index:
        assert ent.index_column == "a"
        assert ent["a"].tolist() == [1, 2, 3]
    else:
        assert ent.index_column == "_ent_index"
        assert ent["_ent_index"].tolist() == [0, 1, 2]


def test_add_emb():
    data = _get_entity_data()

    # Test add embedding columns
    ent = Entity(data)
    embs = np.random.randn(3, 6)
    ent.add_embedding_column("embs", embs)
    embs = torch.randn(3, 6)
    ent.add_embedding_column("embs2", embs)
    embs = EmbeddingColumn(torch.randn(3, 6))
    ent.add_embedding_column("embs3", embs)
    assert ent.embedding_columns == ["embs", "embs2", "embs3"]
    assert isinstance(ent["embs"], EmbeddingColumn)
    assert isinstance(ent["embs2"], EmbeddingColumn)
    assert isinstance(ent["embs3"], EmbeddingColumn)
    assert torch.allclose(ent["embs3"]._data, embs._data)

    # Test length match error
    embs = np.random.randn(4, 6)
    with pytest.raises(
        AssertionError, match="Length of embedding needs to be the same as data"
    ):
        ent.add_embedding_column("embs3", embs)

    # Test index_to_rowid
    embs_data = np.random.randn(4, 6)
    gold_embs = np.zeros((3, 6))
    gold_embs[0] = embs[1]
    gold_embs[1] = embs[0]
    gold_embs[2] = embs[3]

    # Test numpy
    ent = Entity(data, index_column="c")
    embs = embs_data
    ent.add_embedding_column("embs", embs, index_to_rowid={"x": 1, "y": 0, "z": 3})
    assert ent["embs"] == gold_embs

    # Test tensor
    ent = Entity(data, index_column="c")
    embs = torch.as_tensor(embs_data)
    ent.add_embedding_column("embs", embs, index_to_rowid={"x": 1, "y": 0, "z": 3})
    assert ent["embs"] == gold_embs

    # Test embedding column
    ent = Entity(data, index_column="c")
    embs = EmbeddingColumn(embs_data)
    ent.add_embedding_column("embs", embs, index_to_rowid={"x": 1, "y": 0, "z": 3})
    assert ent["embs"] == gold_embs


@pytest.mark.parametrize(
    "k",
    [1, 2],
)
def test_find_similar(k):
    data = _get_entity_data()

    ent = Entity(data, index_column="c", embedding_columns=["g"])
    sim = ent.most_similar("x", k)
    assert isinstance(sim, Entity)
    assert len(sim) == k
    assert sim.column_names == ["a", "b", "c", "d", "e", "f", "g", "index"]
