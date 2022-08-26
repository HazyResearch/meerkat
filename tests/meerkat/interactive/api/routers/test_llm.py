import numpy as np
import pytest
from fastapi.testclient import TestClient

import meerkat as mk
from meerkat.interactive.api.main import app

client = TestClient(app)

def test_get_categories():
    from meerkat.state import state
    state.secrets.add(api="ai21", api_key="")
    # TODO(karan): this .set() below fails to work, 
    # but ideally we would like it to be done here
    # the issue is caused by Manifest, which sets up a SQLLite cache
    # that throws an error that it's being used in multiple threads (and
    # can only be used in the thread in which it was created)
    # As a workaround, we run a set inside the API call at the moment.
    state.llm.set(client="ai21", engine="j1-jumbo")
    response = client.post(
        "/llm/generate/categories",
        json={
            "dataset_description": "face images of people",
            "hint": "i'm interested in exploring unusual correlations between attributes",
        },
    )
    assert response.status_code == 200

def test_get_categorization():
    from meerkat.state import state
    state.secrets.add(api="ai21", api_key="")
    # TODO(karan): this .set() below fails to work, 
    # but ideally we would like it to be done here
    # the issue is caused by Manifest, which sets up a SQLLite cache
    # that throws an error that it's being used in multiple threads (and
    # can only be used in the thread in which it was created)
    # As a workaround, we run a set inside the API call at the moment.
    state.llm.set(client="ai21", engine="j1-jumbo")
    response = client.post(
        "/llm/generate/categorization",
        json={
            "description": "types of paintings",
            "existing_categories": ["none"],
        },
    )
    assert response.status_code == 200
