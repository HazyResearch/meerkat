import numpy as np
import pytest

import meerkat as mk
from meerkat.interactive.app.src.lib.component.core.filter import FilterCriterion

# TODO (arjun): undo the skip filter stuff


@mk.gui.endpoint
def _set_criteria(criteria, store):
    store.set(criteria)


def test_filter_single_criterion():
    df = mk.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    with mk.gui._react():
        filter = mk.gui.Filter(df=df)
        out = filter(df)
    node = out.inode

    assert filter.criteria == []

    criterion = FilterCriterion(is_enabled=True, column="a", op=">", value=5)
    _set_criteria(filter.criteria.value + [criterion], filter.criteria)
    assert np.all(node.obj["a"].data > 5)

    _set_criteria([], filter.criteria)
    assert np.all(node.obj["a"].data == df["a"].data)

    criterion = FilterCriterion(is_enabled=True, column="a", op="==", value=5)
    _set_criteria(filter.criteria.value + [criterion], filter.criteria)
    assert np.all(node.obj["a"].data == 5)


@pytest.mark.parametrize("op", [">", "<", ">=", "<=", "==", "in", "not in"])
@pytest.mark.parametrize("value", [5, [5, 10]])
def test_filter_operations(op, value):
    if "in" not in op and isinstance(value, (list, tuple)):
        # Skip these cases because they are not valid.
        return
    elif "in" in op and not isinstance(value, (list, tuple)):
        value = [value]

    df = mk.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
    if op == "==":
        expected = df["a"] == value
    elif op == ">":
        expected = df["a"] > value
    elif op == "<":
        expected = df["a"] < value
    elif op == ">=":
        expected = df["a"] >= value
    elif op == "<=":
        expected = df["a"] <= value
    elif op == "!=":
        expected = df["a"] != value
    elif op == "in":
        expected = df["a"].data.isin(value)
        value = ",".join([str(v) for v in value])
    elif op == "not in":
        expected = ~df["a"].data.isin(value)
        value = ",".join([str(v) for v in value])
    expected = df[expected]["a"]

    with mk.gui._react():
        filter = mk.gui.Filter(df=df)
        out = filter(df)
    node = out.inode

    criterion = FilterCriterion(is_enabled=True, column="a", op=op, value=value)
    _set_criteria(filter.criteria.value + [criterion], filter.criteria)
    assert np.all(node.obj["a"].data == expected)


def test_filter_bool():
    """TODO: Test filtering with a boolean column."""
    pass


def test_skip_filter_disabled():
    """Test logic for skipping the filter component when adding/modifying
    disabled criteria."""
    df = mk.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    with mk.gui._react():
        filter = mk.gui.Filter(df=df)
        out = filter(df)
    node = out.inode

    # The filter criterion is disabled, so the output dataframe should not change.
    criterion = FilterCriterion(is_enabled=False, column="a", op=">", value=5)
    _set_criteria([criterion], filter.criteria)
    assert id(node.obj) == id(out)

    # The filter criterion is disabled, so changing this criterion should not
    # change the output dataframe.
    criterion.op = "<"
    _set_criteria([criterion], filter.criteria)
    assert id(node.obj) == id(out)

    # Deleting a disabled criterion should not change the output dataframe.
    _set_criteria([], filter.criteria)
    assert id(node.obj) == id(out)


def test_skip_filter_duplicate():
    """If a criterion is added that is a duplicate of an existing criterion, it
    should be skipped."""
    df = mk.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    with mk.gui._react():
        filter = mk.gui.Filter(df=df)
        out = filter(df)
    node = out.inode

    # Duplicate of the same criterion.
    criterion = FilterCriterion(is_enabled=True, column="a", op=">", value=5)
    _set_criteria([criterion], filter.criteria)
    obj_id = id(node.obj)

    duplicate_criterion = FilterCriterion(is_enabled=True, column="a", op=">", value=5)
    _set_criteria([criterion, duplicate_criterion], filter.criteria)
    assert id(node.obj) == obj_id


def test_skip_filter_order():
    """Filter criteria are order-agnostic.

    If the same criteria are added in a different order, the output
    dataframe should not change.
    """
    df = mk.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    with mk.gui._react():
        filter = mk.gui.Filter(df=df)
        out = filter(df)
    node = out.inode

    # Duplicate of the same criterion.
    criteria = [
        FilterCriterion(is_enabled=True, column="a", op=">", value=5),
        FilterCriterion(is_enabled=True, column="a", op="<", value=10),
    ]
    _set_criteria(criteria, filter.criteria)
    obj_id = id(node.obj)

    _set_criteria(criteria[::-1], filter.criteria)
    assert id(node.obj) == obj_id
