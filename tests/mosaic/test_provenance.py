"""Unittests for Datasets."""


import numpy as np

import mosaic as ms
from mosaic.datapanel import DataPanel
from mosaic.provenance import (
    ProvenanceObjNode,
    ProvenanceOpNode,
    capture_provenance,
    provenance,
)


def test_obj_creation():
    dp = DataPanel()
    assert isinstance(dp.node, ProvenanceObjNode)


def test_obj_del():
    dp = DataPanel()
    node = dp.node
    del dp
    assert node.ref() is None


def test_from_batch():

    with provenance():
        dp = DataPanel.from_batch(
            {
                "x": np.arange(4),
            }
        )
    assert isinstance(dp.node.last_parent[0], ProvenanceOpNode)
    assert dp.node.last_parent[0].name == "DataPanel.from_batch"
    assert dp.node.last_parent[0]


def test_from_batch_no_provenance():

    with provenance(enabled=False):
        dp = DataPanel.from_batch(
            {
                "x": np.arange(4),
            }
        )
    assert dp.node.last_parent is None


def test_map():
    with provenance():
        dp1 = DataPanel.from_batch(
            {
                "x": np.arange(4),
            }
        )

        dp2 = dp1.map(lambda x: {"z": x["x"] + 1}, batched=True, batch_size=2)

    assert isinstance(dp2.node.last_parent[0], ProvenanceOpNode)
    assert dp2.node.last_parent[1] == tuple()
    assert dp2.node.last_parent[0].name == "MappableMixin.map"
    assert isinstance(dp1.node.last_parent[0], ProvenanceOpNode)
    assert dp1.node.children[-1][1] == ("self",)
    assert dp1.node.children[-1][0].name == "MappableMixin.map"


@capture_provenance(capture_args=["x"])
def custom_fn(dp1, dp2, x):
    dp3 = ms.concat([dp1, dp2], axis="columns")
    return {"dp": dp3, "x": x}, dp2


def test_custom_fn():

    with provenance():
        dp1 = DataPanel.from_batch(
            {
                "x": np.arange(4),
            }
        )
        dp2 = DataPanel.from_batch(
            {
                "y": np.arange(4),
            }
        )
        d, _ = custom_fn(dp1, dp2, x="abc")
        dp3 = d["dp"]

    assert isinstance(dp3.node.last_parent[0], ProvenanceOpNode)
    assert dp3.node.last_parent[0].name == "custom_fn"
    assert dp3.node.last_parent[1] == (0, "dp")

    assert isinstance(dp1.node.last_parent[0], ProvenanceOpNode)
    assert dp1.node.children[-1][0].name == "custom_fn"
    assert dp1.node.children[-1][1] == ("dp1",)

    assert isinstance(dp2.node.last_parent[0], ProvenanceOpNode)
    assert dp2.node.children[-1][0].name == "custom_fn"
    assert dp2.node.children[-1][1] == ("dp2",)

    custom_op = dp3.node.last_parent[0]
    # test that dp2, which was passed in to `custom_fn`, is not in children
    assert custom_op.children == [
        (dp3.node, (0, "dp")),
        *[(dp3[key].node, (0, "dp", key)) for key in dp3.keys()],
    ]

    custom_op.captured_args["x"] == "abc"


def test_get_provenance():

    with provenance():
        dp1 = DataPanel.from_batch(
            {
                "x": np.arange(4),
            }
        )
        dp2 = DataPanel.from_batch(
            {
                "y": np.arange(4),
            }
        )
        d, _ = custom_fn(dp1, dp2, x="abc")
        dp3 = d["dp"]

    nodes, edges = dp3.get_provenance(include_columns=False, last_parent_only=False)
    assert len(nodes) == 8
    assert sum([isinstance(node, ProvenanceObjNode) for node in nodes]) == 3
    assert sum([isinstance(node, ProvenanceOpNode) for node in nodes]) == 5
    assert len(edges) == 9

    nodes, edges = dp3.get_provenance(include_columns=False, last_parent_only=True)
    assert len(nodes) == 6
    assert sum([isinstance(node, ProvenanceObjNode) for node in nodes]) == 3
    assert sum([isinstance(node, ProvenanceOpNode) for node in nodes]) == 3
    assert len(edges) == 5

    nodes, edges = dp3.get_provenance(include_columns=True, last_parent_only=True)
    assert len(nodes) == 10
    assert sum([isinstance(node, ProvenanceObjNode) for node in nodes]) == 7
    assert sum([isinstance(node, ProvenanceOpNode) for node in nodes]) == 3
    assert len(edges) == 13

    nodes, edges = dp3.get_provenance(include_columns=True, last_parent_only=False)
    assert len(nodes) == 16
    assert sum([isinstance(node, ProvenanceObjNode) for node in nodes]) == 7
    assert sum([isinstance(node, ProvenanceOpNode) for node in nodes]) == 9
    assert len(edges) == 28
