import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive import Pivot
from meerkat.interactive.api.main import app
from meerkat.interactive.graph import Store

client = TestClient(app)


@pytest.fixture
def dp_testbed():
    dp = mk.DataPanel(
        {"a": np.arange(10), "b": np.arange(10, 20), "clip(a)": np.zeros((10, 4))}
    )

    return {"dp": dp}


def test_match(dp_testbed, monkeypatch):
    from meerkat.ops import match

    monkeypatch.setattr(match, "embed", lambda *args, **kwargs: np.zeros((1, 4)))

    dp = dp_testbed["dp"]
    box = Pivot(dp)
    response = client.post(
        f"/ops/{box.id}/match/", json={"input": "a", "query": "this is the query"}
    )

    assert response.status_code == 200


def test_match_col_out(dp_testbed, monkeypatch):
    from meerkat.ops import match

    monkeypatch.setattr(match, "embed", lambda *args, **kwargs: np.zeros((1, 4)))

    store = Store("")
    dp = dp_testbed["dp"]
    box = Pivot(dp)
    response = client.post(
        f"/ops/{box.id}/match/",
        json={"input": "a", "query": "this is the query", "col_out": store.id},
    )

    assert response.status_code == 200
    assert store.value == "_match_a_this is the query"
    assert len(response.json()) == 2
