"""Unittests for Datasets."""
from itertools import product

import numpy as np
import pytest
import torch

from mosaic import DataPanel, EmbeddingColumn, ListColumn, NumpyArrayColumn
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
    ent.add_embedding_column("embs3", embs, overwrite=True)
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


@pytest.mark.parametrize(
    "k",
    [1, 2],
)
def test_find_similar_multiple_columns(k):
    data = _get_entity_data()

    ent = Entity(data, index_column="c", embedding_columns=["g"])
    embs_data = np.random.randn(3, 6)
    ent.add_embedding_column("embs2", embs_data)

    with pytest.raises(
        AssertionError,
        match="Length of search embedding needs to match query embedding",
    ):
        ent.most_similar(
            "x", k, query_embedding_column="g", search_embedding_column="embs2"
        )

    ent = Entity(data, index_column="c", embedding_columns=["g"])
    embs_data = np.random.randn(3, 5)
    ent.add_embedding_column("embs2", embs_data)

    sim = ent.most_similar(
        "x", k, query_embedding_column="g", search_embedding_column="embs2"
    )
    assert isinstance(sim, Entity)
    assert len(sim) == k
    assert sim.column_names == ["a", "b", "c", "d", "e", "f", "g", "index", "embs2"]


def test_convert_entities_to_ids():
    data = _get_entity_data()

    ent = Entity(data, index_column="c", embedding_columns=["g"])

    train_data = DataPanel(
        {
            "col1": NumpyArrayColumn(np.array(["x", "z", "z", "y"])),
            "col2": [["x", "x"], ["y"], ["z", "x", "x"], ["y"]],
        }
    )

    col1 = ent.convert_entities_to_ids(train_data["col1"])
    assert isinstance(col1, NumpyArrayColumn)
    assert all(col1 == np.array([0, 2, 2, 1]))
    col2 = ent.convert_entities_to_ids(train_data["col2"])
    assert isinstance(col2, ListColumn)
    assert col2._data == [[0, 0], [1], [2, 0, 0], [1]]


def test_append_entities():
    data1 = _get_entity_data()
    ent1 = Entity(data1, index_column="c", embedding_columns=["g"])

    # Test append column
    data = {
        "c": ["x", "y", "z"],
        "h": [3, 4, 5],
        "g": np.random.rand(3, 5),
    }
    ent2 = Entity(data, index_column="c", embedding_columns=["g"])

    ent3 = ent1.append(ent2, axis=1, overwrite=True)

    gold_data = {
        "a": [1, 2, 3],
        "b": [True, False, True],
        "c": ["x", "y", "z"],
        "d": [{"e": 2}, {"e": 3}, {"e": 4}],
        "e": torch.ones(3),
        "f": np.ones(3),
        "g": data["g"],
        "h": [3, 4, 5],
    }
    assert ent3.column_names == ["a", "b", "c", "d", "e", "f", "g", "index", "h"]
    assert ent3["h"].tolist() == gold_data["h"]
    assert ent3["c"]._data == gold_data["c"]
    assert ent3.index_column == "c"
    assert ent3.embedding_columns == ["g"]

    # Test append column no index match
    data = {
        "c": ["x", "a", "z"],
        "h": [3, 4, 5],
        "g": np.random.rand(3, 5),
    }
    ent2 = Entity(data, index_column="c", embedding_columns=["g"])
    with pytest.raises(
        ValueError,
        match=r"Can only append along axis 1 \(columns\) if the entity indexes match",
    ):
        _ = ent1.append(ent2, axis=1)

    # Test append rows
    data = {
        "a": [4, 5, 6],
        "b": [True, False, True],
        "c": ["x", "y", "z"],
        "d": [{"e": 2}, {"e": 3}, {"e": 4}],
        "e": torch.ones(3),
        "f": np.ones(3),
        "g": np.random.rand(3, 5),
    }
    ent2 = Entity(data, index_column="c", embedding_columns=["g"])

    ent3 = ent1.append(ent2, axis=1)

    gold_data = {
        "a": [4, 5, 6],
        "b": [True, False, True],
        "c": ["x", "y", "z"],
        "d": [{"e": 2}, {"e": 3}, {"e": 4}],
        "e": torch.ones(3),
        "f": np.ones(3),
        "g": np.random.rand(3, 5),
    }
    assert ent3._data == gold_data
