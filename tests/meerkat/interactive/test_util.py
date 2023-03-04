import numpy as np
import pytest
import torch
from fastapi.encoders import jsonable_encoder

from meerkat.interactive.app.src.lib.component.core.match import MatchCriterion
from meerkat.interactive.graph.store import Store
from meerkat.interactive.utils import get_custom_json_encoder


@pytest.mark.parametrize(
    "obj,expected",
    [
        # torch
        (torch.as_tensor([1, 2, 3]), [1, 2, 3]),
        (torch.as_tensor([1, 2, 3]).float(), [1.0, 2.0, 3.0]),
        # numpy
        (np.array([1, 2, 3]), [1, 2, 3]),
        (np.array([1, 2, 3]).astype(np.float32), [1.0, 2.0, 3.0]),
        (np.array(["foo", "bar"]), ["foo", "bar"]),
        (np.array([[1.0, 2.0, 3.0]]), [[1.0, 2.0, 3.0]]),
        (np.asarray(1.0), 1.0),
        (np.asarray(1).astype(np.float16), 1.0),
        (np.asarray(1).astype(np.float32), 1.0),
        (np.asarray(1).astype(np.float64), 1.0),
        (np.asarray(1).astype(np.int8), 1),
        (np.asarray(1).astype(np.int16), 1),
        (np.asarray(1).astype(np.int32), 1),
        (np.asarray(1).astype(np.int64), 1),
    ],
)
@pytest.mark.parametrize("use_store", [False, True])
def test_custom_json_encoder_native_objects(obj, expected, use_store: bool):
    if use_store:
        obj = Store(obj)

    out = jsonable_encoder(obj, custom_encoder=get_custom_json_encoder())
    np.testing.assert_equal(out, expected)


@pytest.mark.parametrize(
    "obj,expected",
    [
        (
            MatchCriterion(
                against="foo",
                query="my query",
                name="my name",
                query_embedding=np.asarray([1, 2, 3]),
                positives=[1, 2, 3, 4],
                negatives=[5, 6, 7, 8],
            ),
            {
                "against": "foo",
                "query": "my query",
                "name": "my name",
                "query_embedding": [1, 2, 3],
                "positives": [1, 2, 3, 4],
                "negatives": [5, 6, 7, 8],
            },
        )
    ],
)
@pytest.mark.parametrize("use_store", [False, True])
def test_custom_json_encoder_custom_objects(obj, expected, use_store: bool):
    if use_store:
        obj = Store(obj)
    out = jsonable_encoder(obj, custom_encoder=get_custom_json_encoder())
    np.testing.assert_equal(out, expected)


@pytest.mark.parametrize(
    "obj,expected",
    [
        (Store(Store([1, 2, 3])), [1, 2, 3]),
        (Store((Store([1, 2, 3]), Store([4, 5, 6]))), [[1, 2, 3], [4, 5, 6]]),
        (Store({"foo": Store([1, 2, 3])}), {"foo": [1, 2, 3]}),
    ],
)
def test_custom_json_encoder_nested_stores(obj, expected):
    out = jsonable_encoder(obj, custom_encoder=get_custom_json_encoder())
    np.testing.assert_equal(out, expected)
