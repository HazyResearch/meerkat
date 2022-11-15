from typing import List

import numpy as np
import pandas as pd

import meerkat as mk
from meerkat.interactive.graph import (
    Reference,
    ReferenceModification,
    interface_op,
    trigger,
)


@interface_op
def binary_op(df_1: mk.DataFrame, df_2: mk.DataFrame):
    return mk.DataFrame({"a": df_1["a"] + df_2["a"]})


@interface_op
def unary_op(df_1):
    return mk.DataFrame({"a": df_1["a"] * 3})


def test_trigger():
    df_1 = mk.DataFrame({"a": np.arange(10)})
    df_2 = mk.DataFrame({"a": np.arange(10)})

    ref_1 = Reference(df_1)
    ref_2 = Reference(df_2)

    derived_1 = binary_op(ref_1, ref_2)
    derived_2 = unary_op(derived_1)
    derived_3 = binary_op(derived_1, derived_2)
    derived_4 = binary_op(derived_3, ref_2)

    ref_1.obj = mk.DataFrame({"a": np.arange(10, 20)})
    ref_2.obj = mk.DataFrame({"a": np.arange(10, 20)})
    modifications = trigger(
        [
            ReferenceModification(id=ref_1.id, scope=[]),
            ReferenceModification(id=ref_2.id, scope=[]),
        ],
    )

    assert len(modifications) == 6
    assert (derived_1.obj["a"] == np.arange(10, 20) * 2).all()
    assert (derived_2.obj["a"] == derived_1.obj["a"] * 3).all()
    assert (derived_3.obj["a"] == derived_2.obj["a"] + derived_1.obj["a"]).all()
    assert (derived_4.obj["a"] == derived_3.obj["a"] + np.arange(10, 20)).all()


def _create_dummy_df() -> mk.DataFrame:
    df = pd.DataFrame({"a": np.arange(10), "b": np.arange(10) + 10})
    return mk.DataFrame.from_pandas(df)


@interface_op
def _add_to_list(_keys: List[str], new_key: str):
    return _keys + [new_key]


def test_react_context_manager_basic():

    df = _create_dummy_df()

    with mk.gui.react():
        keys_reactive = df.keys()
        _ = _add_to_list(keys_reactive, "c")

    assert isinstance(keys_reactive, mk.gui.Store)
    assert keys_reactive.has_trigger_children()

    # Outside of context manager.
    keys = df.keys()
    assert isinstance(keys, List)


def test_react_context_manager_nested():
    df = _create_dummy_df()

    with mk.gui.react():
        keys_reactive = df.keys()
        with mk.gui.no_react():
            keys = df.keys()

    assert isinstance(keys_reactive, mk.gui.Store)
    assert isinstance(keys, List)
