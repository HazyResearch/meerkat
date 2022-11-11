import numpy as np

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
