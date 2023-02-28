import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive.api.main import app

client = TestClient(app)


@pytest.fixture
def df_testbed():
    df = mk.DataFrame(
        {"a": np.arange(10), "b": np.arange(10, 20), "clip(a)": np.zeros((10, 4))}
    )
    df.set_primary_key("a")

    return {"df": df}


def test_get_schema(df_testbed):
    df: mk.DataFrame = df_testbed["df"]
    response = client.post(
        f"/df/{df.id}/schema/",
        json={"columns": ["a", "b"]},
    )
    assert response.status_code == 200
    assert response.json() == {
        "id": df.id,
        "columns": [
            {
                "name": "a",
                "type": "PandasScalarColumn",
                "cellComponent": "MeerkatNumber",
                "cellProps": {
                    "dtype": "int",
                    "precision": 3,
                    "percentage": False,
                    "classes": "",
                },
                "cellDataProp": "data",
            },
            {
                "name": "b",
                "type": "PandasScalarColumn",
                "cellComponent": "MeerkatNumber",
                "cellProps": {
                    "dtype": "int",
                    "precision": 3,
                    "percentage": False,
                    "classes": "",
                },
                "cellDataProp": "data",
            },
        ],
        "nrows": 10,
        "primaryKey": "a",
    }


def test_rows(df_testbed):
    df: mk.DataFrame = df_testbed["df"]
    response = client.post(
        f"/df/{df.id}/rows/",
        json={"start": 3, "end": 7},
    )
    assert response.status_code == 200

    response_json = response.json()

    assert response_json["columnInfos"] == [
        {
            "name": "a",
            "type": "PandasScalarColumn",
            "cellComponent": "MeerkatNumber",
            "cellProps": {
                "dtype": "int",
                "precision": 3,
                "percentage": False,
                "classes": "",
            },
            "cellDataProp": "data",
        },
        {
            "name": "b",
            "type": "PandasScalarColumn",
            "cellComponent": "MeerkatNumber",
            "cellProps": {
                "dtype": "int",
                "precision": 3,
                "percentage": False,
                "classes": "",
            },
            "cellDataProp": "data",
        },
        {
            "name": "clip(a)",
            "type": "NumPyTensorColumn",
            "cellComponent": "MeerkatTensor",
            "cellProps": {"dtype": "float64"},
            "cellDataProp": "data",
        },
    ]
    assert response_json["rows"] == [
        [3, 13, {"data": [0.0, 0.0, 0.0, 0.0], "shape": [4], "dtype": "float64"}],
        [4, 14, {"data": [0.0, 0.0, 0.0, 0.0], "shape": [4], "dtype": "float64"}],
        [5, 15, {"data": [0.0, 0.0, 0.0, 0.0], "shape": [4], "dtype": "float64"}],
        [6, 16, {"data": [0.0, 0.0, 0.0, 0.0], "shape": [4], "dtype": "float64"}],
    ]
    assert response_json["fullLength"] == 10
    assert response_json["posidxs"] == [3, 4, 5, 6]
    assert response_json["primaryKey"] == df.primary_key


@pytest.mark.skip
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
            "cell_props": {"dtype": "int"},
        },
        {
            "name": "b",
            "type": "NumpyArrayColumn",
            "cell_component": "basic",
            "cell_props": {"dtype": "int"},
        },
        {
            "name": "clip(a)",
            "type": "NumpyArrayColumn",
            "cell_component": "basic",
            "cell_props": {"dtype": "str"},
        },
        {
            "name": "c",
            "type": "NumpyArrayColumn",
            "cell_component": "basic",
            "cell_props": {"dtype": "float"},
        },
    ]


@pytest.mark.skip
@pytest.mark.parametrize("aggregation", ["mean"])
def test_aggregate_w_name(df_testbed, aggregation: str):
    df = df_testbed["df"]
    response = client.post(
        f"/df/{df.id}/aggregate/",
        json={"aggregation": aggregation},
    )

    assert response.status_code == 200
    assert response.json() == {"a": 4.5, "b": 14.5, "clip(a)": 0.0}


@pytest.mark.skip
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


@pytest.mark.skip
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
