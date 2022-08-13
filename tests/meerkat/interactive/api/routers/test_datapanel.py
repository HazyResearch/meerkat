import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive.api.main import app

client = TestClient(app)


@pytest.fixture
def dp_testbed():
    dp = mk.DataPanel({"a": np.arange(10), "b": np.arange(10, 20)})

    return {"dp": dp}


def test_get_schema(dp_testbed):
    dp = dp_testbed["dp"]
    response = client.post(
        f"/dp/{dp.id}/schema/",
        json={"columns": ["a", "b"]},
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": dp.id,
        "columns": [
            {
                "name": "a",
                "type": "NumpyArrayColumn",
                "cell_component": "basic",
                "cell_props": {},
            },
            {
                "name": "b",
                "type": "NumpyArrayColumn",
                "cell_component": "basic",
                "cell_props": {},
            },
        ],
    }


@pytest.mark.parametrize("aggregation", ["mean"])
def test_aggregate_w_name(dp_testbed, aggregation: str):
    dp = dp_testbed["dp"]
    response = client.post(
        f"/dp/{dp.id}/aggregate/",
        json={"aggregation": aggregation},
    )

    assert response.status_code == 200
    assert response.json() == {"a": 4.5, "b": 14.5}


def test_aggregate_w_id_accepts_dp(dp_testbed):
    dp = dp_testbed["dp"]

    from meerkat.interactive.gui import Aggregation

    aggregation = lambda dp: (dp["a"] + dp["b"]).mean()  # noqa: E731
    aggregation = Aggregation(aggregation)

    response = client.post(
        f"/dp/{dp.id}/aggregate/",
        json={"aggregation_id": aggregation.id, "accepts_dp": True},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"dp": np.mean(dp["b"] + dp["a"])}


def test_aggregate_w_id_accepts_col(dp_testbed):
    dp = dp_testbed["dp"]

    from meerkat.interactive.gui import Aggregation

    aggregation = lambda col: col.mean()  # noqa: E731
    aggregation = Aggregation(aggregation)

    response = client.post(
        f"/dp/{dp.id}/aggregate/",
        json={
            "aggregation_id": aggregation.id,
            "columns": ["a"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"a": np.mean(dp["a"])}
