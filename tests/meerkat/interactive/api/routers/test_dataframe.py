import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive import Pivot
from meerkat.interactive.api.main import app
from meerkat.interactive.edit import EditTargetConfig

client = TestClient(app)


@pytest.fixture
def df_testbed():
    df = mk.DataFrame(
        {"a": np.arange(10), "b": np.arange(10, 20), "clip(a)": np.zeros((10, 4))}
    )

    return {"df": df}


def test_get_schema(df_testbed):
    df = df_testbed["df"]
    box = Pivot(df)
    response = client.post(
        f"/df/{box.id}/schema/",
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


def test_rows(df_testbed):
    df = df_testbed["df"]
    box = Pivot(df)
    response = client.post(
        f"/df/{box.id}/rows/",
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
    df = mk.DataFrame(
        {
            "row_id": column_type(list(map(str, np.arange(10, 20)))),
            "value": column_type(list(map(str, np.arange(10)))),
        }
    )
    df.data.consolidate()
    pivot = Pivot(df)

    response = client.post(
        f"/df/{pivot.id}/edit/",
        json={"value": "100", "column": "value", "row_id": "14", "id_column": "row_id"},
    )
    assert response.status_code == 200
    assert df["value"][4] == "100"
    assert response.json() == [{"id": pivot.id, "scope": ["value"], "type": "box"}]


@pytest.mark.parametrize("column_type", [mk.PandasSeriesColumn])
def test_edit_target(column_type):
    df = mk.DataFrame(
        {
            "row_id_s": column_type(list(map(str, np.arange(10, 20)))),
            "value": column_type(list(map(str, np.arange(10)))),
        }
    )

    target_df = mk.DataFrame(
        {
            "row_id_t": column_type(list(map(str, np.arange(0, 20)))),
            "value": column_type(list(map(str, np.arange(0, 20)))),
        }
    )

    df.data.consolidate()
    pivot = Pivot(df)
    target_pivot = Pivot(target_df)

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
    response = client.post(f"/df/{pivot.id}/edit_target/", json=data)

    assert response.status_code == 200, response.json()
    assert target_df["value"][15] == "100"
    assert target_df["value"][16] == "100"
    assert target_df["value"][18] == "100"


@pytest.mark.parametrize("column_type", [mk.PandasSeriesColumn])
def test_edit_target_keys(column_type):
    df = mk.DataFrame(
        {
            "row_id_s": column_type(list(map(str, np.arange(10, 20)))),
            "value": column_type(list(map(str, np.arange(10)))),
        }
    )

    target_df = mk.DataFrame(
        {
            "row_id_t": column_type(list(map(str, np.arange(0, 20)))),
            "value": column_type(list(map(str, np.arange(0, 20)))),
        }
    )

    df.data.consolidate()
    pivot = Pivot(df)
    target_pivot = Pivot(target_df)

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
    response = client.post(f"/df/{pivot.id}/edit_target/", json=data)

    assert response.status_code == 200, response.json()
    assert target_df["value"][15] == "100"
    assert target_df["value"][16] == "100"
    assert target_df["value"][18] == "100"


def test_remove_row_by_index(df_testbed):
    df = df_testbed["df"]

    pivot = Pivot(df)
    data = {
        "row_index": "5",
    }

    response = client.post(f"/df/{pivot.id}/remove_row_by_index/", json=data)

    assert response.status_code == 200, response.json()


@pytest.mark.parametrize("column_type", [mk.PandasSeriesColumn])
def test_edit_target_missing_id(column_type):
    df = mk.DataFrame(
        {
            "row_id_s": column_type(list(map(str, np.arange(0, 10)))),
            "value": column_type(list(map(str, np.arange(10)))),
        }
    )

    target_df = mk.DataFrame(
        {
            "row_id_t": column_type(list(map(str, np.arange(5, 20)))),
            "value": column_type(list(map(str, np.arange(5, 20)))),
        }
    )

    df.data.consolidate()
    pivot = Pivot(df)
    target_pivot = Pivot(target_df)

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
    response = client.post(f"/df/{pivot.id}/edit_target/", json=data)

    assert response.status_code == 500, response.json()


def test_add(df_testbed):
    df = df_testbed["df"]
    df = Pivot(df)
    response = client.post(
        f"/df/{df.id}/add/",
        json={"column": "z"},
    )
    assert response.status_code == 200


def test_sort(df_testbed):
    df = df_testbed["df"]
    df["c"] = np.random.rand(10)
    response = client.post(f"/df/{df.id}/sort/", json={"by": "c"})
    assert response.status_code == 200
    assert response.json()["id"] != df.id
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
def test_aggregate_w_name(df_testbed, aggregation: str):
    df = df_testbed["df"]
    response = client.post(
        f"/df/{df.id}/aggregate/",
        json={"aggregation": aggregation},
    )

    assert response.status_code == 200
    assert response.json() == {"a": 4.5, "b": 14.5}


def test_aggregate_w_id_accepts_df(df_testbed):
    df = df_testbed["df"]

    from meerkat.interactive.gui import Aggregation

    aggregation = lambda df: (df["a"] + df["b"]).mean()  # noqa: E731
    aggregation = Aggregation(aggregation)

    response = client.post(
        f"/df/{df.id}/aggregate/",
        json={"aggregation_id": aggregation.id, "accepts_df": True},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"df": np.mean(df["b"] + df["a"])}


def test_aggregate_w_id_accepts_col(df_testbed):
    df = df_testbed["df"]

    from meerkat.interactive.gui import Aggregation

    aggregation = lambda col: col.mean()  # noqa: E731
    aggregation = Aggregation(aggregation)

    response = client.post(
        f"/df/{df.id}/aggregate/",
        json={
            "aggregation_id": aggregation.id,
            "columns": ["a"],
        },
    )

    assert response.status_code == 200, response.text
    assert response.json() == {"a": np.mean(df["a"])}
