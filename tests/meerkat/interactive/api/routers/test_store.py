import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive import Pivot
from meerkat.interactive.api.main import app
from meerkat.interactive.graph import (
    BoxModification,
    Pivot,
    Store,
    interface_op,
    trigger,
)

client = TestClient(app)


@pytest.fixture
def dp_testbed():
    dp = mk.DataPanel(
        {"a": np.arange(10), "b": np.arange(10, 20), "clip(a)": np.zeros((10, 4))}
    )

    return {"dp": dp}


def test_store():

    store = Store(0)
    derived = unary_op(store)
    response = client.post(f"/store/{store.id}/trigger/", json={"value": 2})

    assert response.status_code == 200
    assert derived.obj == 5
    assert store.value == 2


@interface_op
def unary_op(value):
    return value + 3
