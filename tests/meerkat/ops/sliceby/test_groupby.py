import numpy as np

from meerkat import NumPyTensorColumn, ObjectColumn
from meerkat.dataframe import DataFrame
from meerkat.ops.sliceby.groupby import GroupBy, groupby

# Comment for meeting 5/19: Testing group by multiple columns,
# single columns on list, on string.

# Different columns as by: including ListColumn, NumpyArrayColumn,
# TensorColumn

# Key index is NumpyArrays, TensorColumn

# Coverage: 100%


def assertNumpyArrayEquality(arr1, arr2):
    assert np.allclose(arr1, arr2)


def test_group_by_type():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(
                    ["sam", "liam", "sam", "owen", "liam", "connor", "connor"],
                    dtype=str,
                )
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, "name")

    assert isinstance(df, GroupBy)


def test_tensor_column_by():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, "a")
    out = df["b"].mean()

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_integer_type():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, "a")
    out = df["b"].mean()

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_integer_type_md():
    b = np.zeros((7, 4))
    b[0, 0] = 4
    b[1, 1] = 3
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn(b),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = df.groupby("a")
    out = df["b"].mean(axis=0)

    assertNumpyArrayEquality(out["b"][0], np.array([2, 0, 0, 0]))
    assert out["a"][0] == 1
    assert out["a"][1] == 2
    assert out["a"][2] == 3

    assertNumpyArrayEquality(out["b"][1], np.array([0, 1, 0, 0]))
    assertNumpyArrayEquality(out["b"][2], np.array([0, 0, 0, 0]))


def test_group_by_integer_type_axis_passed():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, "a")
    out = df["b"].mean(axis=0)

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_integer_type_as_prop():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = df.groupby("a")
    out = df["b"].mean()

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_tensor_key():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, "a")
    out = df["b"].mean()

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_string_type():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, "name")
    out = df["b"].mean()
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumPyTensorColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_string_type_multiple_keys():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, "name")
    out = df[["b", "a"]].mean()
    assert (np.linalg.norm(out["a"].data - np.array([1.5, 2.5, 1, 2.5]))) < 1e-10
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumPyTensorColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_by_string_type_as_list():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, ["name"])
    out = df["b"].mean()
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumPyTensorColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_by_string_type_as_list_key_as_list():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, ["name"])
    out = df[["b"]].mean()
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumPyTensorColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_float_should_fail():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )
    try:
        groupby(df, ["c"])
        assert False
    except Exception:
        assert True


def test_group_by_float_should_fail_nonexistent_column():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )
    try:
        groupby(df, ["d"])
        assert False
    except Exception:
        assert True


def test_group_by_by_string_type_as_list_key_as_list_mult_key_by_name():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, ["name"])
    out = df[["b", "c"]].mean()
    assert np.linalg.norm(out["c"].data - np.array([1.55, 3.75, 4.3, 7.05]) < 1e-10)
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumPyTensorColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_by_string_type_as_list_key_as_list_mult_key():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "a_diff": NumPyTensorColumn([1, 2, 2, 2, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, ["a", "a_diff"])
    out = df[["b"]].mean()
    assertNumpyArrayEquality(out["a"].data, np.array([1, 1, 2, 3]))
    assertNumpyArrayEquality(out["a_diff"].data, np.array([1, 2, 2, 3]))
    assertNumpyArrayEquality(out["b"].data, np.array([1, 4, 11.0 / 3.0, 6]))


def test_group_by_by_string_type_as_list_key_as_list_mult_key_tensor():
    df = DataFrame(
        {
            "a": NumPyTensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "a_diff": NumPyTensorColumn([1, 2, 2, 2, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, ["a", "a_diff"])
    out = df[["b"]].mean()
    assertNumpyArrayEquality(out["a"].data, np.array([1, 1, 2, 3]))
    assertNumpyArrayEquality(out["a_diff"].data, np.array([1, 2, 2, 3]))
    assertNumpyArrayEquality(out["b"].data, np.array([1, 4, 11.0 / 3.0, 6]))


def test_simple_list_column():
    df = DataFrame(
        {
            "a": ObjectColumn([1, 2, 2, 1, 3, 2, 3]),
            "a_diff": NumPyTensorColumn([1, 2, 2, 2, 3, 2, 3]),
            "name": NumPyTensorColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumPyTensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumPyTensorColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(df, ["a", "a_diff"])
    out = df[["b"]].mean()
    assertNumpyArrayEquality(out["a"].data, np.array([1, 1, 2, 3]))
    assertNumpyArrayEquality(out["a_diff"].data, np.array([1, 2, 2, 3]))
    assertNumpyArrayEquality(out["b"].data, np.array([1, 4, 11.0 / 3.0, 6]))
