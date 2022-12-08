import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive.api.main import app
from meerkat.interactive.graph import Store

client = TestClient(app)


@pytest.fixture
def df_testbed():
    df = mk.DataFrame(
        {"a": np.arange(10), "b": np.arange(10, 20), "clip(a)": np.zeros((10, 4))}
    )

    return {"df": df}


def test_match(df_testbed, monkeypatch):
    from meerkat.ops import match

    monkeypatch.setattr(match, "embed", lambda *args, **kwargs: np.zeros((1, 4)))

    df = df_testbed["df"]
    ref = Reference(df)
    response = client.post(
        f"/ops/{ref.id}/match/", json={"input": "a", "query": "this is the query"}
    )
    assert response.status_code == 200


def test_match_col_out(df_testbed, monkeypatch):
    from meerkat.ops import match

    monkeypatch.setattr(match, "embed", lambda *args, **kwargs: np.zeros((1, 4)))

    store = Store("")
    df = df_testbed["df"]
    ref = Reference(df)
    response = client.post(
        f"/ops/{ref.id}/match/",
        json={"input": "a", "query": "this is the query", "col_out": store.id},
    )

    assert response.status_code == 200
    assert store.value == "_match_a_this is the query"
    assert len(response.json()) == 2
