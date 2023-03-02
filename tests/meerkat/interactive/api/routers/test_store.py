import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive.api.main import app
from meerkat.interactive.graph import Store, reactive

client = TestClient(app)


@pytest.fixture
def df_testbed():
    df = mk.DataFrame(
        {"a": np.arange(10), "b": np.arange(10, 20), "clip(a)": np.zeros((10, 4))}
    )

    return {"df": df}


def test_store():
    store = Store(0)
    derived = unary_op(store)
    response = client.post(f"/store/{store.id}/update/", json={"value": 2})

    assert response.status_code == 200
    assert derived.value == 5
    assert store.value == 2


@reactive
def unary_op(value):
    return value + 3
