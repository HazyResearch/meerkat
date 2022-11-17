from typing import List

import numpy as np
import pandas as pd

import meerkat as mk
from meerkat.interactive.graph import (
    reactive,
    trigger,
)
from meerkat.interactive.modification import DataFrameModification


@reactive
def binary_op(df_1: mk.DataFrame, df_2: mk.DataFrame):
    return mk.DataFrame({"a": df_1["a"] + df_2["a"]})


@reactive
def unary_op(df_1):
    return mk.DataFrame({"a": df_1["a"] * 3})


def test_trigger():
    # FIXME: fix this test
    df_1 = mk.DataFrame({"a": np.arange(10)})
    df_2 = mk.DataFrame({"a": np.arange(10)})

    derived_1 = binary_op(df_1, df_2)
    derived_2 = unary_op(derived_1)
    derived_3 = binary_op(derived_1, derived_2)
    derived_4 = binary_op(derived_3, df_2)

    df_1 = mk.DataFrame({"a": np.arange(10, 20)})
    df_2 = mk.DataFrame({"a": np.arange(10, 20)})
    modifications = trigger(
        [
            DataFrameModification(id=df_1.inode.id, scope=[]),
            DataFrameModification(id=df_2.inode.id, scope=[]),
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


@reactive
def _add_to_list(_keys: List[str], new_key: str):
    return _keys + [new_key]


def test_react_context_manager_basic():

    df = _create_dummy_df()

    with mk.gui.react():
        keys_reactive = df.keys()
        _ = _add_to_list(keys_reactive, "c")

    assert isinstance(keys_reactive, mk.gui.Store)
    assert keys_reactive.inode.has_trigger_children()

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


def test_default_nested_return():
    """By default, nested return is True for functions returning tuples."""
    @reactive
    def _return_tuple():
        return ("a", "b")

    @reactive
    def _return_list():
        return ["a", "b"]

    with mk.gui.react():
        out = _return_tuple()
        a, b = out
    assert not isinstance(out, mk.gui.Store)
    assert isinstance(a, mk.gui.Store)
    assert isinstance(b, mk.gui.Store)

    with mk.gui.react():
        out = _return_list()
    assert isinstance(out, mk.gui.Store)
