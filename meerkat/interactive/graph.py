from copy import copy
from dataclasses import dataclass
from functools import wraps
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Union

from pydantic import BaseModel

from meerkat.mixins.identifiable import IdentifiableMixin

if TYPE_CHECKING:
    from meerkat.datapanel import DataPanel


class BoxConfig(BaseModel):
    box_id: str
    type: str = "DataPanel"


class Box(IdentifiableMixin):
    identifiable_group: str = "boxes"

    def __init__(self, obj):
        super().__init__()
        self.obj = obj

        self.children: List[Operation] = []

    @property
    def config(self):
        return BoxConfig(box_id=self.id, type="DataPanel")


class Modification(BaseModel):
    box_id: str
    scope: List[str]

    @property
    def box(self):
        from meerkat.state import state

        return state.identifiables.get(group="boxes", id=self.box_id)


class PivotConfig(BoxConfig):
    pass


class Pivot(Box):
    pass


class DerivedConfig(BoxConfig):
    pass


class Derived(Box):
    pass


class StoreConfig(BaseModel):
    store_id: str
    value: Any


class Store(IdentifiableMixin):
    identifiable_group: str = "stores"

    def __init__(self, value: Any):
        super().__init__()
        self.value = value

    @property
    def config(self):
        return StoreConfig(
            store_id=self.id,
            value=self.value,
        )


Storeable = Union[int, str, float]


def make_store(value: Union[str, Storeable]):
    if isinstance(value, Store):
        return value
    return Store(value)


@dataclass
class Operation:
    fn: Callable
    args: List[Any]
    kwargs: Dict[str, Any]
    result: Derived

    def __call__(self):
        unpacked_args, unpacked_kwargs, _ = _unpack_boxes(*self.args, **self.kwargs)
        result = self.fn(*unpacked_args, **unpacked_kwargs)
        # TODO: result may be multiple
        self.result.obj = result
        return self.result


def trigger(modifications: List[Modification]) -> List[Modification]:
    modifications = copy(modifications)
    all_modifications = []
    while modifications:
        modification = modifications.pop(0)
        all_modifications.append(modification)
        box = modification.box
        for op in box.children:
            derived = op()
            modifications.append(Modification(box_id=derived.id, scope=[]))

    # ops_to_exec, modifications = topological(pivots)

    # for op in ops_to_exec:
    #     op()

    return all_modifications


def _unpack_boxes(*args, **kwargs):
    # TODO(Sabri): this should be nested
    boxes = []
    unpacked_args = []
    for arg in args:
        if isinstance(arg, Box):
            boxes.append(arg)
            unpacked_args.append(arg.obj)
        else:
            unpacked_args.append(arg)

    unpacked_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, Box):
            boxes.append(v)
            unpacked_kwargs[k] = v.obj
        else:
            unpacked_kwargs[k] = v

    return unpacked_args, unpacked_kwargs, boxes


def interface_op(fn: Callable):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # TODO(Sabri): this should be nested
        unpacked_args, unpacked_kwargs, boxes = _unpack_boxes(*args, **kwargs)

        result = fn(*unpacked_args, **unpacked_kwargs)

        if len(boxes) > 0:
            derived = Derived(result)
            op = Operation(fn=fn, args=args, kwargs=kwargs, result=derived)
            for input_box in boxes:
                input_box.children.append(op)
            # TODO(Sabri): this should be nested
            return derived

        return result

    return wrapper


@interface_op
def head(dp: "DataPanel", n: int = 5):
    new_dp = dp.head(n)
    import numpy as np

    new_dp["head_column"] = np.zeros(len(new_dp))
    return new_dp
