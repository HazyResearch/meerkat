import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive.api.main import app
from meerkat.ops.sliceby.explainby import ExplainBy

client = TestClient(app)


@pytest.fixture
def sliceby_testbed():
    df = mk.DataFrame(
        {
            "a": [1, 2, 2, 1, 3, 2, 3],
            "name": np.array(
                ["sam", "liam", "sam", "owen", "liam", "connor", "connor"],
                dtype=str,
            ),
            "b": [1, 2, 3, 4, 5, 6, 7],
            "c": [1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6],
        }
    )
    gb = df.groupby(by="name")

    return {"df": df, "sb": gb}


@pytest.fixture
def explainby_testbed():
    df = mk.DataFrame(
        {
            "a": [1, 2, 2, 1, 3, 2, 3],
            "name": np.array(
                ["sam", "liam", "sam", "owen", "liam", "connor", "connor"],
                dtype=str,
            ),
            "b": [1, 2, 3, 4, 5, 6, 7],
            "c": [1.0, 3.2, 2.1, 4.3, 5.4, 6.5, 7.6],
        }
    )
    eb = ExplainBy(
        data=df,
        by="name",
        scores={
            0: [1, 2, 3, 4, 5, 6, 7],
            1: [7, 6, 5, 4, 3, 2, 1],
        },
    )
    return {"df": df, "sb": eb}


def test_rows_explainby(explainby_testbed):
    sb = explainby_testbed["sb"]
    response = client.post(
        f"/sliceby/{sb.id}/rows/",
        json={"slice_key": 0, "start": 0, "end": 2, "columns": ["b"]},
    )
    assert response.status_code == 200, response.text
    assert response.json() == {
        "column_infos": [
            {
                "name": "b",
                "type": "NumpyArrayColumn",
                "cell_component": "basic",
                "cell_props": {},
            }
        ],
        "indices": [0, 1],
        "rows": [[" 7"], [" 6"]],
        "full_length": 7,
    }


@pytest.mark.parametrize("aggregation", ["mean"])
def test_aggregate_w_name(sliceby_testbed, aggregation: str):
    sb = sliceby_testbed["sb"]
    response = client.post(
        f"/sliceby/{sb.id}/aggregate/",
        json={"aggregation": aggregation},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "a": {"connor": " 2.5", "liam": " 2.5", "owen": " 1.0", "sam": " 1.5"},
        "b": {"connor": " 6.5", "liam": " 3.5", "owen": " 4.0", "sam": " 2.0"},
        "c": {"connor": " 7.05", "liam": " 4.3", "owen": " 4.3", "sam": " 1.55"},
    }


@pytest.mark.parametrize("aggregation", ["mean"])
def test_aggregate_w_name_w_columns(sliceby_testbed, aggregation: str):
    sb = sliceby_testbed["sb"]
    response = client.post(
        f"/sliceby/{sb.id}/aggregate/",
        json={"aggregation": aggregation, "columns": ["a"]},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "a": {"connor": " 2.5", "liam": " 2.5", "owen": " 1.0", "sam": " 1.5"}
    }


def test_aggregate_w_id(sliceby_testbed):
    sb = sliceby_testbed["sb"]

    from meerkat.interactive.gui import Aggregation

    aggregation = lambda df: (df["a"] + df["b"]).mean()  # noqa: E731
    aggregation = Aggregation(aggregation)

    response = client.post(
        f"/sliceby/{sb.id}/aggregate/",
        json={"aggregation_id": aggregation.id, "accepts_df": True},
    )

    assert response.status_code == 200, response.text
    assert response.json() == {
        "df": {"connor": " 9.0", "liam": " 6.0", "owen": " 5.0", "sam": " 3.5"}
    }
