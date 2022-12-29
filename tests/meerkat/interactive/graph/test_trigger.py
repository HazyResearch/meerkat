import numpy as np

import meerkat as mk
from meerkat.interactive.graph import trigger
from meerkat.interactive.modification import DataFrameModification
from meerkat.state import state


@mk.gui.reactive
def binary_op(df_1: mk.DataFrame, df_2: mk.DataFrame) -> mk.DataFrame:
    return mk.DataFrame({"a": df_1["a"] + df_2["a"]})


@mk.gui.reactive
def unary_op(df_1) -> mk.DataFrame:
    return mk.DataFrame({"a": df_1["a"] * 3})


@mk.gui.endpoint
def update_df(df: mk.DataFrame, col: str, value: np.ndarray) -> mk.DataFrame:
    df[col] = value
    return df


def test_trigger():
    df_1 = mk.DataFrame({"a": np.arange(10)})
    df_2 = mk.DataFrame({"a": np.arange(10)})

    with mk.gui.react():
        derived_1 = binary_op(df_1, df_2)
        derived_2 = unary_op(derived_1)
        derived_3 = binary_op(derived_1, derived_2)
        derived_4 = binary_op(derived_3, df_2)

    # Unpack the node from the output dataframes.
    derived_1_node = derived_1.inode
    derived_2_node = derived_2.inode
    derived_3_node = derived_3.inode
    derived_4_node = derived_4.inode

    # Update the values of df_1 and df_2.
    update_df(df_1, "a", np.arange(10, 20))
    update_df(df_2, "a", np.arange(10, 20))

    # TODO: Figure out why we need to force add modifications.
    # Shouldn't calling the endpoint automatically add these modifications?
    state.modification_queue.ready()
    modifications = [
        DataFrameModification(id=df_1.inode.id, scope=[]),
        DataFrameModification(id=df_2.inode.id, scope=[]),
    ]
    for mod in modifications:
        state.modification_queue.add(mod)
    modifications = trigger()
    state.modification_queue.unready()

    # The node is attached to different dataframes on trigger.
    # So we need to fetch the updated dataframe associated with the node.
    derived_1 = derived_1_node.obj
    derived_2 = derived_2_node.obj
    derived_3 = derived_3_node.obj
    derived_4 = derived_4_node.obj

    assert len(modifications) == 6
    assert (derived_1["a"] == np.arange(10, 20) * 2).all()
    assert (derived_2["a"] == derived_1["a"] * 3).all()
    assert (derived_3["a"] == derived_2["a"] + derived_1["a"]).all()
    assert (derived_4["a"] == derived_3["a"] + np.arange(10, 20)).all()
