from typing import List, Union

import meerkat as mk


def make_test_df(
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
):
    """Helper function, returns test df."""
    df = mk.DataFrame(
        {
            "tensor": mk.TorchTensorColumn([3, 1, 2]),
            "pandas": mk.ScalarColumn([9, 8, 7]),
            "numpy": mk.TorchTensorColumn([5, 4, 6]),
        }
    )
    test = df.sort(by=by, ascending=ascending)
    return test


def make_tiebreaker_test_df(
    by: Union[str, List[str]],
    ascending: Union[bool, List[bool]] = True,
):
    df = mk.DataFrame(
        {
            "tensor": mk.TorchTensorColumn([3, 2, 1]),
            "pandas": mk.ScalarColumn([9, 7, 9]),
            "numpy": mk.TorchTensorColumn([4, 4, 6]),
        }
    )
    test = df.sort(by=by, ascending=ascending)
    return test


# flake8: noqa
######## SINGLE COLUMN TESTS ########


def test_sort_by_ascending_tensor_column():
    """Testing all columns after sorting by an ascending tensor column."""

    test = make_test_df(by=["tensor"])
    assert (
        (test["tensor"] == mk.TorchTensorColumn([1, 2, 3])).all()
        and (test["pandas"] == mk.ScalarColumn([8, 7, 9])).all()
        and (test["numpy"] == mk.TorchTensorColumn([4, 6, 5])).all()
    )


def test_sort_by_ascending_pandas_on_pandas_column():
    """Testing all columns after sorting by an ascending pandas column."""

    test = make_test_df(by=["pandas"])
    assert (
        (test["tensor"] == mk.TorchTensorColumn([2, 1, 3])).all()
        and (test["pandas"] == mk.ScalarColumn([7, 8, 9])).all()
        and (test["numpy"] == mk.TorchTensorColumn([6, 4, 5])).all()
    )


def test_sort_single_numpy_column_ascending():
    """Testing all columns after sorting by an ascending numpy column."""

    test = make_test_df(by=["numpy"])
    assert (
        (test["tensor"] == mk.TorchTensorColumn([1, 3, 2])).all()
        and (test["pandas"] == mk.ScalarColumn([8, 9, 7])).all()
        and (test["numpy"] == mk.TorchTensorColumn([4, 5, 6])).all()
    )


# flake8: noqa
######## SINGLE COLUMN TESTS DESCENDING ########


def test_sort_single_tensor_column_descending():
    """Testing all columns after sorting by a descending tensor column."""

    test = make_test_df(by=["tensor"], ascending=False)
    assert (
        (test["tensor"] == mk.TorchTensorColumn([3, 2, 1])).all()
        and (test["pandas"] == mk.ScalarColumn([9, 7, 8])).all()
        and (test["numpy"] == mk.TorchTensorColumn([5, 6, 4])).all()
    )


def test_sort_single_pandas_column_descending():
    """Testing all columns after sorting by a descending pandas column."""
    test = make_test_df(by=["pandas"], ascending=False)
    assert (
        (test["tensor"] == mk.TorchTensorColumn([3, 1, 2])).all()
        and (test["pandas"] == mk.ScalarColumn([9, 8, 7])).all()
        and (test["numpy"] == mk.TorchTensorColumn([5, 4, 6])).all()
    )


def test_sort_single_numpy_column_descending():
    """Testing all columns after sorting by a descending numpy column."""
    test = make_test_df(by=["numpy"], ascending=False)
    assert (
        (test["tensor"] == mk.TorchTensorColumn([2, 3, 1])).all()
        and (test["pandas"] == mk.ScalarColumn([7, 9, 8])).all()
        and (test["numpy"] == mk.TorchTensorColumn([6, 5, 4])).all()
    )


######## MULTIPLE COLUMN TESTS ########


def test_sort_numpy_and_tensor_ascending():
    """# Testing all columns after sorting with multiple ascending columns
    (numpy and tensor)"""
    test = make_tiebreaker_test_df(by=["numpy", "tensor"], ascending=True)
    assert (
        (test["tensor"] == mk.TorchTensorColumn([2, 3, 1])).all()
        and (test["pandas"] == mk.ScalarColumn([7, 9, 9])).all()
        and (test["numpy"] == mk.TorchTensorColumn([4, 4, 6])).all()
    )


def test_sort_numpy_and_pandas_ascending():
    """Testing all columns after sorting with multiple ascending columns (numpy
    and tensor)"""
    test = make_tiebreaker_test_df(by=["numpy", "pandas"], ascending=True)
    assert (
        (test["tensor"] == mk.TorchTensorColumn([2, 3, 1])).all()
        and (test["pandas"] == mk.ScalarColumn([7, 9, 9])).all()
        and (test["numpy"] == mk.TorchTensorColumn([4, 4, 6])).all()
    )


def test_sort_numpy_and_pandas_ascending_variable():
    """Testing all columns after sorting with multiple ascending columns (numpy
    and tensor)"""
    test = make_tiebreaker_test_df(by=["numpy", "pandas"], ascending=[True, False])
    assert (
        (test["tensor"] == mk.TorchTensorColumn([3, 2, 1])).all()
        and (test["pandas"] == mk.ScalarColumn([9, 7, 9])).all()
        and (test["numpy"] == mk.TorchTensorColumn([4, 4, 6])).all()
    )


def test_sort_numpy_and_pandas_and_tensor_ascending():
    """Testing all columns after sorting with multiple ascending columns (numpy
    and pandas and tensor)"""
    df = mk.DataFrame(
        {
            "tensor": mk.TorchTensorColumn([3, 2, 1]),
            "pandas": mk.ScalarColumn([9, 7, 7]),
            "numpy": mk.TorchTensorColumn([6, 4, 4]),
        }
    )
    test = df.sort(by=["numpy", "pandas", "tensor"], ascending=True)
    assert (
        (test["tensor"] == mk.TorchTensorColumn([1, 2, 3])).all()
        and (test["pandas"] == mk.ScalarColumn([7, 7, 9])).all()
        and (test["numpy"] == mk.TorchTensorColumn([4, 4, 6])).all()
    )


def test_sort_tensor_and_pandas_descending():
    """Testing all columns after sorting with multiple ascending columns
    (tensor and pandas)."""
    df = mk.DataFrame(
        {
            "tensor": mk.TorchTensorColumn([3, 2, 2]),
            "pandas": mk.ScalarColumn([9, 8, 7]),
            "numpy": mk.TorchTensorColumn([6, 4, 4]),
        }
    )
    test = df.sort(by=["tensor", "pandas"], ascending=False)
    assert (
        (test["tensor"] == mk.TorchTensorColumn([3, 2, 2])).all()
        and (test["pandas"] == mk.ScalarColumn([9, 8, 7])).all()
        and (test["numpy"] == mk.TorchTensorColumn([6, 4, 4])).all()
    )
