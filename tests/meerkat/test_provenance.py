"""Unittests for Datasets."""


import numpy as np

import meerkat as mk
from meerkat.dataframe import DataFrame
from meerkat.provenance import (
    ProvenanceObjNode,
    ProvenanceOpNode,
    capture_provenance,
    provenance,
)


def test_obj_creation():
    df = DataFrame()
    assert isinstance(df.node, ProvenanceObjNode)


def test_obj_del():
    df = DataFrame()
    node = df.node
    del df
    assert node.ref() is None


def test_map():
    with provenance():
        df1 = DataFrame.from_batch(
            {
                "x": np.arange(4),
            }
        )

        df2 = df1.map(lambda x: {"z": x["x"] + 1}, is_batched_fn=True, batch_size=2)

    assert isinstance(df2.node.last_parent[0], ProvenanceOpNode)
    assert df2.node.last_parent[1] == tuple()
    assert df2.node.last_parent[0].name == "DataFrame.map"
    assert isinstance(df1.node.last_parent[0], ProvenanceOpNode)
    assert df1.node.children[-1][1] == ("self",)
    assert df1.node.children[-1][0].name == "DataFrame.map"


@capture_provenance(capture_args=["x"])
def custom_fn(df1, df2, x):
    df3 = mk.concat([df1, df2], axis="columns")
    return {"df": df3, "x": x}, df2


def test_custom_fn():
    with provenance():
        df1 = DataFrame.from_batch(
            {
                "x": np.arange(4),
            }
        )
        df2 = DataFrame.from_batch(
            {
                "y": np.arange(4),
            }
        )
        d, _ = custom_fn(df1, df2, x="abc")
        df3 = d["df"]

    assert isinstance(df3.node.last_parent[0], ProvenanceOpNode)
    assert df3.node.last_parent[0].name == "custom_fn"
    assert df3.node.last_parent[1] == (0, "df")

    assert isinstance(df1.node.last_parent[0], ProvenanceOpNode)
    assert df1.node.children[-1][0].name == "custom_fn"
    assert df1.node.children[-1][1] == ("df1",)

    assert isinstance(df2.node.last_parent[0], ProvenanceOpNode)
    assert df2.node.children[-1][0].name == "custom_fn"
    assert df2.node.children[-1][1] == ("df2",)

    custom_op = df3.node.last_parent[0]
    # test that df2, which was passed in to `custom_fn`, is not in children
    # also that the columns, which are unchanged, are nto in children
    assert custom_op.children == [
        (df3.node, (0, "df")),
    ]

    custom_op.captured_args["x"] == "abc"


def test_get_provenance():
    with provenance():
        df1 = DataFrame.from_batch(
            {
                "x": np.arange(4),
            }
        )
        df2 = DataFrame.from_batch(
            {
                "y": np.arange(4),
            }
        )
        d, _ = custom_fn(df1, df2, x="abc")
        df3 = d["df"]

    nodes, edges = df3.get_provenance(include_columns=False, last_parent_only=False)
    assert len(nodes) == 7
    assert sum([isinstance(node, ProvenanceObjNode) for node in nodes]) == 3
    assert sum([isinstance(node, ProvenanceOpNode) for node in nodes]) == 4
    assert len(edges) == 8

    nodes, edges = df3.get_provenance(include_columns=False, last_parent_only=True)
    assert len(nodes) == 6
    assert sum([isinstance(node, ProvenanceObjNode) for node in nodes]) == 3
    assert sum([isinstance(node, ProvenanceOpNode) for node in nodes]) == 3
    assert len(edges) == 5

    nodes, edges = df3.get_provenance(include_columns=True, last_parent_only=True)
    assert len(nodes) == 8
    assert sum([isinstance(node, ProvenanceObjNode) for node in nodes]) == 5
    assert sum([isinstance(node, ProvenanceOpNode) for node in nodes]) == 3
    assert len(edges) == 9

    nodes, edges = df3.get_provenance(include_columns=True, last_parent_only=False)
    assert len(nodes) == 9
    assert sum([isinstance(node, ProvenanceObjNode) for node in nodes]) == 5
    assert sum([isinstance(node, ProvenanceOpNode) for node in nodes]) == 4
    assert len(edges) == 14


def test_repr():
    with provenance():
        df1 = DataFrame.from_batch(
            {
                "x": np.arange(4),
            }
        )
        df2 = DataFrame.from_batch(
            {
                "y": np.arange(4),
            }
        )
        d, _ = custom_fn(df1, df2, x="abc")
        df3 = d["df"]

    assert repr(df1) == "DataFrame(nrows: 4, ncols: 1)"
    assert repr(df2) == "DataFrame(nrows: 4, ncols: 1)"
    assert repr(df3) == "DataFrame(nrows: 4, ncols: 2)"
