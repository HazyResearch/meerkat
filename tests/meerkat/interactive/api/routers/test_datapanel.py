import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive import Pivot
from meerkat.interactive.api.main import app
from meerkat.interactive.edit import EditTargetConfig

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


@pytest.mark.parametrize("column_type", [mk.PandasSeriesColumn])
def test_edit(column_type):
    dp = mk.DataPanel(
        {
            "row_id": column_type(list(map(str, np.arange(10, 20)))),
            "value": column_type(list(map(str, np.arange(10)))),
        }
    )
    dp.data.consolidate()
    pivot = Pivot(dp)

    response = client.post(
        f"/dp/{pivot.id}/edit/",
        json={"value": "100", "column": "value", "row_id": "14", "id_column": "row_id"},
    )
    assert response.status_code == 200
    assert dp["value"][4] == "100"
    assert response.json() == [{"id": pivot.id, "scope": ["value"], "type": "box"}]


@pytest.mark.parametrize("column_type", [mk.PandasSeriesColumn])
def test_edit_target(column_type):
    dp = mk.DataPanel(
        {
            "row_id_s": column_type(list(map(str, np.arange(10, 20)))),
            "value": column_type(list(map(str, np.arange(10)))),
        }
    )

    target_dp = mk.DataPanel(
        {
            "row_id_t": column_type(list(map(str, np.arange(0, 20)))),
            "value": column_type(list(map(str, np.arange(0, 20)))),
        }
    )

    dp.data.consolidate()
    pivot = Pivot(dp)
    target_pivot = Pivot(target_dp)

    data = {
        "target": EditTargetConfig(
            target=target_pivot.config,
            target_id_column="row_id_t",
            source_id_column="row_id_s",
        ).dict(),
        "value": "100",
        "column": "value",
        "row_indices": [5, 6, 8],
    }
    response = client.post(f"/dp/{pivot.id}/edit_target/", json=data)

    assert response.status_code == 200, response.json()
    assert target_dp["value"][15] == "100"
    assert target_dp["value"][16] == "100"
    assert target_dp["value"][18] == "100"


@pytest.mark.parametrize("column_type", [mk.PandasSeriesColumn])
def test_edit_target_keys(column_type):
    dp = mk.DataPanel(
        {
            "row_id_s": column_type(list(map(str, np.arange(10, 20)))),
            "value": column_type(list(map(str, np.arange(10)))),
        }
    )

    target_dp = mk.DataPanel(
        {
            "row_id_t": column_type(list(map(str, np.arange(0, 20)))),
            "value": column_type(list(map(str, np.arange(0, 20)))),
        }
    )

    dp.data.consolidate()
    pivot = Pivot(dp)
    target_pivot = Pivot(target_dp)

    data = {
        "target": EditTargetConfig(
            target=target_pivot.config,
            target_id_column="row_id_t",
            source_id_column="row_id_s",
        ).dict(),
        "value": "100",
        "column": "value",
        "row_keys": [15, 16, 18],
        "primary_key": "row_id_s",
    }
    response = client.post(f"/dp/{pivot.id}/edit_target/", json=data)

    assert response.status_code == 200, response.json()
    assert target_dp["value"][15] == "100"
    assert target_dp["value"][16] == "100"
    assert target_dp["value"][18] == "100"


@pytest.mark.parametrize("column_type", [mk.PandasSeriesColumn])
def test_edit_target_missing_id(column_type):
    dp = mk.DataPanel(
        {
            "row_id_s": column_type(list(map(str, np.arange(0, 10)))),
            "value": column_type(list(map(str, np.arange(10)))),
        }
    )

    target_dp = mk.DataPanel(
        {
            "row_id_t": column_type(list(map(str, np.arange(5, 20)))),
            "value": column_type(list(map(str, np.arange(5, 20)))),
        }
    )

    dp.data.consolidate()
    pivot = Pivot(dp)
    target_pivot = Pivot(target_dp)

    data = {
        "target": EditTargetConfig(
            target=target_pivot.config,
            target_id_column="row_id_t",
            source_id_column="row_id_s",
        ).dict(),
        "value": "100",
        "column": "value",
        "row_indices": [2, 6, 8],
    }
    response = client.post(f"/dp/{pivot.id}/edit_target/", json=data)

    assert response.status_code == 500, response.json()


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
