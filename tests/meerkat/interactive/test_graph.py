from ast import unaryop
from graphlib import TopologicalSorter

import numpy as np
import pytest

import meerkat as mk
from meerkat.interactive.graph import BoxModification, Pivot, interface_op, trigger


@interface_op
def binary_op(dp_1: mk.DataPanel, dp_2: mk.DataPanel):
    return mk.DataPanel({"a": dp_1["a"] + dp_2["a"]})


@interface_op
def unary_op(dp_1):
    return mk.DataPanel({"a": dp_1["a"] * 3})


def test_trigger():
    dp_1 = mk.DataPanel({"a": np.arange(10)})
    dp_2 = mk.DataPanel({"a": np.arange(10)})

    box_1 = Pivot(dp_1)
    box_2 = Pivot(dp_2)

    derived_1 = binary_op(box_1, box_2)
    derived_2 = unary_op(derived_1)
    derived_3 = binary_op(derived_1, derived_2)
    derived_4 = binary_op(derived_3, box_2)

    box_1.obj = mk.DataPanel({"a": np.arange(10, 20)})
    box_2.obj = mk.DataPanel({"a": np.arange(10, 20)})
    modifications = trigger(
        [
            BoxModification(id=box_1.id, scope=[]),
            BoxModification(id=box_2.id, scope=[]),
        ],
    )

    assert len(modifications) == 4
    assert (derived_1.obj["a"] == np.arange(10, 20) * 2).all()
    assert (derived_2.obj["a"] == derived_1.obj["a"] * 3).all()
    assert (derived_3.obj["a"] == derived_2.obj["a"] + derived_1.obj["a"]).all()
    assert (derived_4.obj["a"] == derived_3.obj["a"] + np.arange(10, 20)).all()
