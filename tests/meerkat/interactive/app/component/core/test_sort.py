import numpy as np

import meerkat as mk
from meerkat.interactive.app.src.lib.component.core.sort import SortCriterion


@mk.gui.endpoint
def _set_criteria(criteria, store: mk.gui.Store):
    store.set(criteria)


def test_sort_single_criterion():
    """Sorting should work with a single criterion."""
    arr = np.arange(10)
    np.random.shuffle(arr)

    df = mk.DataFrame({"a": arr})

    with mk.gui.reactive():
        sort = mk.gui.Sort(df=df)
        out = sort(df)
    node = out.inode

    # Even without a criterion, the output dataframe should be a view
    # of the input dataframe.
    assert id(out) != id(df)

    criterion = SortCriterion(id="foo", is_enabled=True, column="a", ascending=True)
    _set_criteria([criterion], sort.criteria)
    assert np.all(node.obj["a"].data == np.arange(10))


def test_sort_multiple_criteria():
    a, b = np.arange(10), np.arange(10)
    np.random.shuffle(a)
    np.random.shuffle(b)

    df = mk.DataFrame({"a": a, "b": b})

    with mk.gui.reactive():
        sort = mk.gui.Sort(df=df)
        out = sort(df)
    node = out.inode

    # Sort with a.
    criteria = [
        SortCriterion(id="foo", is_enabled=True, column="a", ascending=True),
        SortCriterion(id="foo", is_enabled=True, column="b", ascending=True),
    ]
    _set_criteria(criteria, sort.criteria)
    assert np.all(node.obj["a"].data == np.arange(10))

    # Sort with b.
    criteria = [
        SortCriterion(id="foo", is_enabled=True, column="a", ascending=True),
        SortCriterion(id="foo", is_enabled=True, column="b", ascending=True),
    ]
    _set_criteria(criteria[::-1], sort.criteria)
    assert np.all(node.obj["b"].data == np.arange(10))


def test_skip_sort_disabled():
    """If a criterion is disabled, it should be skipped."""
    df = mk.DataFrame({"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})

    with mk.gui.reactive():
        sort = mk.gui.Sort(df=df)
        out = sort(df)
    node = out.inode

    # The sort criterion is disabled, no output dataframe should not change.
    criterion = SortCriterion(id="foo", is_enabled=False, column="a", ascending=True)
    _set_criteria([criterion], sort.criteria)
    assert id(node.obj) == id(out)

    _set_criteria([], sort.criteria)
    assert id(node.obj) == id(out)

    # The sort criterion is enabled, so the dataframe should change.
    criterion.is_enabled = True
    _set_criteria([criterion], sort.criteria)
    assert id(node.obj) != id(out)


def test_skip_sort_order():
    """When the order of the sort criteria changes, the output dataframe should
    change."""
    df = mk.DataFrame(
        {"a": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], "b": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
    )

    with mk.gui.reactive():
        sort = mk.gui.Sort(df=df)
        out = sort(df)
    node = out.inode

    criteria = [
        SortCriterion(id="foo", is_enabled=True, column="a", ascending=True),
        SortCriterion(id="foo", is_enabled=True, column="b", ascending=False),
    ]
    _set_criteria(criteria, sort.criteria)
    out_id = id(node.obj)

    _set_criteria(criteria[::-1], sort.criteria)
    assert id(node.obj) != out_id
