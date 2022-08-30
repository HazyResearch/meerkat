import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive import Pivot
from meerkat.interactive.api.main import app

client = TestClient(app)


@pytest.fixture
def dp_testbed():
    dp = mk.DataPanel(
        {"a": np.arange(10), "b": np.arange(10, 20), "clip(a)": np.zeros((10, 4))}
    )

    return {"dp": dp}


def test_get_schema(dp_testbed):
    dp = dp_testbed["dp"]
    box = Pivot(dp)
    response = client.post(
        f"/dp/{box.id}/schema/",
        json={"columns": ["a", "b"]},
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": box.obj.id,
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
        "nrows": 10,
    }


def test_rows(dp_testbed):
    dp = dp_testbed["dp"]
    box = Pivot(dp)
    response = client.post(
        f"/dp/{box.id}/rows/",
        json={"start": 3, "end": 7},
    )
    assert response.status_code == 200
    assert response.json()["rows"] == [
        [" 3", " 13", "[0. 0. 0. 0.]"],
        [" 4", " 14", "[0. 0. 0. 0.]"],
        [" 5", " 15", "[0. 0. 0. 0.]"],
        [" 6", " 16", "[0. 0. 0. 0.]"],
    ]
    assert response.json()["indices"] == [3, 4, 5, 6]
    assert response.json()["full_length"] == 10


def test_add(dp_testbed):
    dp = dp_testbed["dp"]
    dp = Pivot(dp)
    response = client.post(
        f"/dp/{dp.id}/add/",
        json={"column": "z"},
    )
    assert response.status_code == 200


def test_sort(dp_testbed):
    dp = dp_testbed["dp"]
    dp["c"] = np.random.rand(10)
    response = client.post(f"/dp/{dp.id}/sort/", json={"by": "c"})
    assert response.status_code == 200
    assert response.json()["id"] != dp.id
    assert response.json()["columns"] == [
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
        {
            "name": "c",
            "type": "NumpyArrayColumn",
            "cell_component": "basic",
            "cell_props": {},
        },
    ]


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
