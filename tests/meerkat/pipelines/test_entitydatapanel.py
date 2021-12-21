"""Unittests for Datasets."""
from itertools import product

import numpy as np
import pytest
import torch

from meerkat import DataPanel, ListColumn, NumpyArrayColumn
from meerkat.ml import EmbeddingColumn
from meerkat.pipelines.entitydatapanel import EntityDataPanel


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
            ent = EntityDataPanel(data, embedding_columns=["g"], index_column="a")
        else:
            ent = EntityDataPanel(data, embedding_columns=["g"])
    else:
        if with_index:
            ent = EntityDataPanel(data, index_column="a")
        else:
            ent = EntityDataPanel(data)

    if with_embs:
        assert ent._embedding_columns == ["g"]
        assert isinstance(ent["g"], EmbeddingColumn)
    else:
        assert ent._embedding_columns == []

    if with_index:
        assert ent._index_column == "a"
        assert ent["a"].tolist() == [1, 2, 3]
    else:
        assert ent._index_column == "_ent_index"
        assert ent["_ent_index"].tolist() == [0, 1, 2]


def test_add_emb():
    data = _get_entity_data()

    # Test add embedding columns
    ent = EntityDataPanel(data)
    embs = np.random.randn(3, 6)
    ent.add_embedding_column("embs", embs)
    embs = torch.randn(3, 6)
    ent.add_embedding_column("embs2", embs)
    embs = EmbeddingColumn(torch.randn(3, 6))
    ent.add_embedding_column("embs3", embs)
    ent.add_embedding_column("embs3", embs, overwrite=True)
    assert ent._embedding_columns == ["embs", "embs2", "embs3"]
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
    ent = EntityDataPanel(data, index_column="c")
    embs = embs_data
    ent.add_embedding_column("embs", embs, index_to_rowid={"x": 1, "y": 0, "z": 3})
    assert ent["embs"] == gold_embs

    # Test tensor
    ent = EntityDataPanel(data, index_column="c")
    embs = torch.as_tensor(embs_data)
    ent.add_embedding_column("embs", embs, index_to_rowid={"x": 1, "y": 0, "z": 3})
    assert ent["embs"] == gold_embs

    # Test embedding column
    ent = EntityDataPanel(data, index_column="c")
    embs = EmbeddingColumn(embs_data)
    ent.add_embedding_column("embs", embs, index_to_rowid={"x": 1, "y": 0, "z": 3})
    assert ent["embs"] == gold_embs


@pytest.mark.parametrize(
    "k",
    [1, 2],
)
def test_find_similar(k):
    return  # TODO: reactivate this test
    data = _get_entity_data()

    ent = EntityDataPanel(data, index_column="c", embedding_columns=["g"])
    sim = ent.most_similar("x", k)
    assert isinstance(sim, EntityDataPanel)
    assert len(sim) == k
    assert sim.columns == ["a", "b", "c", "d", "e", "f", "g"]


@pytest.mark.parametrize(
    "k",
    [1, 2],
)
def test_find_similar_multiple_columns(k):
    return  # TODO: reactivate this test
    data = _get_entity_data()

    ent = EntityDataPanel(data, index_column="c", embedding_columns=["g"])
    embs_data = np.random.randn(3, 6)
    ent.add_embedding_column("embs2", embs_data)

    with pytest.raises(
        AssertionError,
        match="Length of search embedding needs to match query embedding",
    ):
        ent.most_similar(
            "x", k, query_embedding_column="g", search_embedding_column="embs2"
        )

    ent = EntityDataPanel(data, index_column="c", embedding_columns=["g"])
    embs_data = np.random.randn(3, 5)
    ent.add_embedding_column("embs2", embs_data)

    sim = ent.most_similar(
        "x", k, query_embedding_column="g", search_embedding_column="embs2"
    )
    assert isinstance(sim, EntityDataPanel)
    assert len(sim) == k
    assert sim.columns == ["a", "b", "c", "d", "e", "f", "g", "embs2"]


def test_convert_entities_to_ids():
    data = _get_entity_data()

    ent = EntityDataPanel(data, index_column="c", embedding_columns=["g"])

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
    ent1 = EntityDataPanel(data1, index_column="c", embedding_columns=["g"])

    # Test append column
    data = {
        "c": ["x", "y", "z"],
        "h": [3, 4, 5],
        "g": np.random.rand(3, 5),
    }
    ent2 = EntityDataPanel(data, index_column="c", embedding_columns=["g"])

    with pytest.raises(
        AssertionError,
        match="Suffixes must be tuple of len 2 when columns share names",
    ):
        ent1.append(ent2, axis=1, overwrite=False)
    ent3 = ent1.append(ent2, axis=1, overwrite=True, suffixes=("_x", "_y"))

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
    assert ent3.columns == ["a", "b", "c", "d", "e", "f", "g", "h"]
    assert ent3["h"].tolist() == gold_data["h"]
    assert (ent3["c"]._data == gold_data["c"]).all()
    assert ent3._index_column == "c"
    assert ent3._embedding_columns == ["g"]

    # Test append column no index match
    data = {
        "c": ["x", "a", "z"],
        "h": [3, 4, 5],
        "g": np.random.rand(3, 5),
    }
    ent2 = EntityDataPanel(data, index_column="c", embedding_columns=["g"])
    with pytest.raises(
        ValueError,
        match=r"Can only append along axis 1 \(columns\) if the entity indexes match",
    ):
        _ = ent1.append(ent2, axis=1)

    # Test append rows
    data = {
        "a": [4, 5, 6],
        "b": [True, False, True],
        "c": ["a", "b", "c"],
        "d": [{"e": 2}, {"e": 3}, {"e": 4}],
        "e": torch.ones(3),
        "f": np.ones(3),
        "g": np.random.rand(3, 5),
    }
    ent2 = EntityDataPanel(data, index_column="c", embedding_columns=["g"])

    ent3 = ent1.append(ent2, axis=0)
    gold_data = {
        "a": [1, 2, 3, 4, 5, 6],
        "b": [True, False, True, True, False, True],
        "c": ["x", "y", "z", "a", "b", "c"],
        "d": [{"e": 2}, {"e": 3}, {"e": 4}, {"e": 2}, {"e": 3}, {"e": 4}],
        "e": torch.ones(6),
        "f": np.ones(6),
        "g": np.concatenate([ent1["g"].numpy(), ent2["g"].numpy()]),
    }
    for c in ["a", "b", "c", "d", "e", "f", "g"]:
        if c in ["a", "b", "c", "d"]:  # if gold is not torch or numpy
            assert [i for i in ent3[c]] == gold_data[c]
        else:
            assert ent3[c] == gold_data[c]


def test_merge_entities():
    data1 = _get_entity_data()
    ent1 = EntityDataPanel(data1, index_column="c", embedding_columns=["g"])

    # Test append column with same index column
    data = {
        "c": ["y", "x", "z"],
        "h": [4, 3, 5],
        "g": np.random.rand(3, 5),
    }
    ent2 = EntityDataPanel(data, index_column="c", embedding_columns=["g"])
    ent3 = ent1.merge(ent2)
    gold_data = {
        "a": [1, 2, 3],
        "b": [True, False, True],
        "c": ["x", "y", "z"],
        "d": [{"e": 2}, {"e": 3}, {"e": 4}],
        "e": torch.ones(3),
        "f": np.ones(3),
        "g_x": data1["g"],
        "g_y": data["g"],
        "h": [3, 4, 5],
    }

    assert set(ent3.columns) == set(
        [
            "a",
            "b",
            "d",
            "e",
            "f",
            "g_x",
            "h",
            "g_y",
            "c",
        ]
    )
    assert ent3._index_column == "c"
    assert ent3._embedding_columns == ["g_x", "g_y"]
    for c in ["a", "b", "c", "d", "e", "f", "g_x", "g_y", "h"]:
        if c in ["a", "b", "c", "d", "h"]:  # if gold is not torch or numpy
            assert [i for i in ent3[c]] == gold_data[c]
        else:
            assert ent3[c] == gold_data[c]

    # Test append column with left getting index on same index column
    data = {
        "c": ["y", "x", "z"],
        "h": ["x", "y", "z"],
        "g": np.random.rand(3, 5),
    }
    ent2 = EntityDataPanel(data, index_column="c", embedding_columns=["g"])
    ent3 = ent2.merge(ent1, left_on="h")
    gold_data = {
        "a": [1, 2, 3],
        "b": [True, False, True],
        "c_x": ["y", "x", "z"],
        "c_y": ["x", "y", "z"],
        "d": [{"e": 2}, {"e": 3}, {"e": 4}],
        "e": torch.ones(3),
        "f": np.ones(3),
        "g_x": data1["g"],
        "g_y": data["g"],
        "h": ["x", "y", "z"],
    }

    assert set(ent3.columns) == set(
        [
            "c_x",
            "h",
            "g_x",
            "a",
            "b",
            "c_y",
            "d",
            "e",
            "f",
            "g_y",
        ]
    )
    assert ent3._index_column == "c_x"
    assert ent3._embedding_columns == ["g_x", "g_y"]
    for c in ["a", "b", "c_x", "c_y", "d", "e", "f", "g_x", "g_y", "h"]:
        if c in ["a", "b", "c_x", "c_y", "d", "h"]:  # if gold is not torch or numpy
            assert [i for i in ent3[c]] == gold_data[c]
        else:
            assert ent3[c] == gold_data[c]

    # Test append column different index
    data = {
        "d": ["y", "x", "z"],
        "h": [4, 3, 5],
        "g": np.random.rand(3, 5),
    }
    ent2 = EntityDataPanel(data, index_column="d", embedding_columns=["g"])

    ent3 = ent1.merge(ent2)
    gold_data = {
        "a": [1, 2, 3],
        "b": [True, False, True],
        "c": ["x", "y", "z"],
        "d_x": [{"e": 2}, {"e": 3}, {"e": 4}],
        "d_y": ["x", "y", "z"],
        "e": torch.ones(3),
        "f": np.ones(3),
        "g_x": data1["g"],
        "g_y": data["g"],
        "h": [3, 4, 5],
    }

    assert set(ent3.columns) == set(
        [
            "a",
            "b",
            "c",
            "d_x",
            "e",
            "f",
            "g_x",
            "d_y",
            "h",
            "g_y",
        ]
    )
    assert ent3._index_column == "c"
    assert ent3._embedding_columns == ["g_x", "g_y"]
    for c in ["a", "b", "c", "d_x", "d_y", "e", "f", "g_x", "g_y", "h"]:
        if c in ["a", "b", "c", "d_x", "d_y", "h"]:  # if gold is not torch or numpy
            assert [i for i in ent3[c]] == gold_data[c]
        else:
            assert ent3[c] == gold_data[c]

    # Test append column with missing index and index column on ent1 in data for ent2
    data = {
        "d": ["x", "a", "z"],
        "c": [3, 4, 5],
        "i": np.random.rand(3, 5),
    }

    ent2 = EntityDataPanel(data, index_column="d", embedding_columns=["i"])
    ent3 = ent1.merge(ent2)
    gold_data = {
        "a": [1, 3],
        "b": [True, True],
        "c_x": ["x", "z"],
        "d_x": [{"e": 2}, {"e": 4}],
        "d_y": ["x", "z"],
        "e": torch.ones(2),
        "f": np.ones(2),
        "g": np.concatenate(
            [data1["g"][0].reshape(1, 5), data1["g"][2].reshape(1, 5)], axis=0
        ),
        "i": np.concatenate(
            [data["i"][0].reshape(1, 5), data["i"][2].reshape(1, 5)], axis=0
        ),
        "c_y": [3, 5],
    }
    assert set(ent3.columns) == set(
        [
            "a",
            "b",
            "c_x",
            "d_x",
            "e",
            "f",
            "g",
            "d_y",
            "c_y",
            "i",
        ]
    )
    assert ent3._index_column == "c_x"
    assert ent3._embedding_columns == ["g", "i"]
    for c in ["a", "b", "c_x", "c_y", "d_x", "d_y", "e", "f", "g", "i"]:
        if c in [
            "a",
            "b",
            "c_x",
            "c_y",
            "d_x",
            "d_y",
            "h",
        ]:  # if gold is not torch or numpy
            assert [i for i in ent3[c]] == gold_data[c]
        else:
            assert ent3[c] == gold_data[c]
