from functools import wraps
from itertools import product
from typing import Any, Dict, List, Type, Union

import numpy as np
import pytest


def column_parametrize(
    testbed_classes: List[Union[Type, Dict]],
    config: dict = None,
    single: bool = False,
):
    params = [
        c.get_params(config=config, single=single) if isinstance(c, type) else c
        for c in testbed_classes
    ]
    return {
        "params": sum([p["argvalues"] for p in params], []),
        "ids": sum([p["ids"] for p in params], []),
    }


@pytest.fixture
def testbed(request, tmpdir):
    testbed_class, config = request.param
    return testbed_class(**config, tmpdir=tmpdir)


def obj_to_id(obj: Any):
    return str(obj)


class AbstractColumnTestBed:

    DEFAULT_CONFIG = {}

    # subclasses can add pytest marks which will be applied to all
    # tests using the testbed
    marks: pytest.Mark = None

    @classmethod
    def get_params(
        cls, config: dict = None, params: dict = None, single: bool = False
    ) -> Dict[str, Any]:
        updated_config = cls.DEFAULT_CONFIG.copy()
        if config is not None:
            updated_config.update(config)
        configs = [
            (cls, config)
            if cls.marks is None
            else pytest.param((cls, config), marks=cls.marks)
            for config in map(
                dict,
                product(*[[(k, v) for v in vs] for k, vs in updated_config.items()]),
            )
        ]
        if single:
            configs = configs[:1]

        if params is None:
            return {
                "argnames": "testbed",
                "argvalues": configs,
                "ids": [str(config) for config in configs],
            }
        else:
            argvalues = list(product(configs, *params.values()))
            return {
                "argnames": "testbed," + ",".join(params.keys()),
                "argvalues": argvalues,
                "ids": [",".join(map(str, values)) for values in argvalues],
            }

    @classmethod
    @wraps(pytest.mark.parametrize)
    def parametrize(
        cls,
        config: dict = None,
        params: dict = None,
        single: bool = False,
    ):

        return pytest.mark.parametrize(
            **cls.get_params(config=config, single=single), indirect=["testbed"]
        )

    @classmethod
    def single(cls, tmpdir):
        return cls(**cls.get_params(single=True)["argvalues"][0][1], tmpdir=tmpdir)

    def get_map_spec(self, key: str = "default"):
        raise NotImplementedError()

    def get_data(self, index):
        raise NotImplementedError()

    def get_data_to_set(self, index):
        # only mutable columns need implement this
        pass

    @staticmethod
    def assert_data_equal(data1: np.ndarray, data2: np.ndarray):
        raise NotImplementedError()
