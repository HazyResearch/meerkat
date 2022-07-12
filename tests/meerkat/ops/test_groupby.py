import numpy as np

from meerkat import ListColumn, NumpyArrayColumn, TensorColumn
from meerkat.datapanel import DataPanel
from meerkat.ops.groupby import GroupBy, groupby

# Comment for meeting 5/19: Testing group by multiple columns,
# single columns on list, on string.

# Different columns as by: including ListColumn, NumpyArrayColumn,
# TensorColumn

# Key index is NumpyArrays, TensorColumn

# Coverage: 100%


def assertNumpyArrayEquality(arr1, arr2):
    assert np.allclose(arr1, arr2)


def test_group_by_type():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(
                    ["sam", "liam", "sam", "owen", "liam", "connor", "connor"],
                    dtype=str,
                )
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, "name")

    assert isinstance(df, GroupBy)


def test_tensor_column_by():
    dp = DataPanel(
        {
            "a": TensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, "a")
    out = df["b"].mean()

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_integer_type():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, "a")
    out = df["b"].mean()

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_integer_type_md():
    b = np.zeros((7, 4))
    b[0, 0] = 4
    b[1, 1] = 3
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn(b),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = dp.groupby("a")
    out = df["b"].mean(axis=0)

    assertNumpyArrayEquality(out["b"][0], np.array([2, 0, 0, 0]))
    assert out["a"][0] == 1
    assert out["a"][1] == 2
    assert out["a"][2] == 3

    assertNumpyArrayEquality(out["b"][1], np.array([0, 1, 0, 0]))
    assertNumpyArrayEquality(out["b"][2], np.array([0, 0, 0, 0]))


def test_group_by_integer_type_md_axis1():
    b = np.zeros((7, 4))
    b[0, 0] = 4
    b[1, 1] = 3
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn(b),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = dp.groupby("a")
    out = df["b"].mean(axis=1)

    assertNumpyArrayEquality(out["b"][0], np.array([1, 0]))
    assert out["a"][0] == 1
    assert out["a"][1] == 2
    assert out["a"][2] == 3

    assertNumpyArrayEquality(out["b"][1], np.array([0.75, 0, 0]))
    assertNumpyArrayEquality(out["b"][2], np.array([0, 0]))


def test_group_by_integer_type_md_no_axis():
    b = np.zeros((7, 4))
    b[0, 0] = 4
    b[1, 1] = 3
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn(b),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = dp.groupby("a")
    out = df["b"].mean(axis=1)
    assert out["b"][0] == 0.5
    assert out["a"][0] == 1
    assert out["a"][1] == 2
    assert out["a"][2] == 3
    assert out["b"][1] == 0.25
    assert out["b"][2] == 0


def test_group_by_integer_type_axis_passed():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, "a")
    out = df["b"].mean(axis=0)

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_integer_type_as_prop():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = dp.groupby("a")
    out = df["b"].mean()

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_tensor_key():

    dp = DataPanel(
        {
            "a": TensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": TensorColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, "a")
    out = df["b"].mean()

    assertNumpyArrayEquality(out["b"].data, np.array([2.5, (2 + 3 + 6) / 3, 6]))
    assertNumpyArrayEquality(out["a"].data, np.array([1, 2, 3]))


def test_group_by_string_type():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, "name")
    out = df["b"].mean()
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumpyArrayColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_string_type_multiple_keys():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, "name")
    out = df[["b", "a"]].mean()
    assert (np.linalg.norm(out["a"].data - np.array([1.5, 2.5, 1, 2.5]))) < 1e-10
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumpyArrayColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_by_string_type_as_list():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, ["name"])
    out = df["b"].mean()
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumpyArrayColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_by_string_type_as_list_key_as_list():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, ["name"])
    out = df[["b"]].mean()
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumpyArrayColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_float_should_fail():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )
    try:
        groupby(dp, ["c"])
        assert False
    except Exception:
        assert True


def test_group_by_float_should_fail_nonexistent_column():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )
    try:
        groupby(dp, ["d"])
        assert False
    except Exception:
        assert True


def test_group_by_by_string_type_as_list_key_as_list_mult_key_by_name():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, ["name"])
    out = df[["b", "c"]].mean()
    assert np.linalg.norm(out["c"].data - np.array([1.55, 3.75, 4.3, 7.05]) < 1e-10)
    assertNumpyArrayEquality(out["b"].data, np.array([2, 3.5, 4, 6.5]))
    assert (out["name"].data == NumpyArrayColumn(["a", "b", "c", "d"]).data).all()


def test_group_by_by_string_type_as_list_key_as_list_mult_key():
    dp = DataPanel(
        {
            "a": NumpyArrayColumn([1, 2, 2, 1, 3, 2, 3]),
            "a_diff": NumpyArrayColumn([1, 2, 2, 2, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, ["a", "a_diff"])
    out = df[["b"]].mean()
    assertNumpyArrayEquality(out["a"].data, np.array([1, 1, 2, 3]))
    assertNumpyArrayEquality(out["a_diff"].data, np.array([1, 2, 2, 3]))
    assertNumpyArrayEquality(out["b"].data, np.array([1, 4, 11.0 / 3.0, 6]))


def test_group_by_by_string_type_as_list_key_as_list_mult_key_tensor():
    dp = DataPanel(
        {
            "a": TensorColumn([1, 2, 2, 1, 3, 2, 3]),
            "a_diff": NumpyArrayColumn([1, 2, 2, 2, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, ["a", "a_diff"])
    out = df[["b"]].mean()
    assertNumpyArrayEquality(out["a"].data, np.array([1, 1, 2, 3]))
    assertNumpyArrayEquality(out["a_diff"].data, np.array([1, 2, 2, 3]))
    assertNumpyArrayEquality(out["b"].data, np.array([1, 4, 11.0 / 3.0, 6]))


def test_simple_list_column():
    dp = DataPanel(
        {
            "a": ListColumn([1, 2, 2, 1, 3, 2, 3]),
            "a_diff": NumpyArrayColumn([1, 2, 2, 2, 3, 2, 3]),
            "name": NumpyArrayColumn(
                np.array(["a", "b", "a", "c", "b", "d", "d"], dtype=str)
            ),
            "b": NumpyArrayColumn([1, 2, 3, 4, 5, 6, 7]),
            "c": NumpyArrayColumn([1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6]),
        }
    )

    df = groupby(dp, ["a", "a_diff"])
    out = df[["b"]].mean()
    assertNumpyArrayEquality(out["a"].data, np.array([1, 1, 2, 3]))
    assertNumpyArrayEquality(out["a_diff"].data, np.array([1, 2, 2, 3]))
    assertNumpyArrayEquality(out["b"].data, np.array([1, 4, 11.0 / 3.0, 6]))
