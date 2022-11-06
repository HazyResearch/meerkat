from typing import Any, Callable, Dict, List, Union

from meerkat.columns.abstract import AbstractColumn
from meerkat.dataframe import DataFrame
from meerkat.mixins.identifiable import IdentifiableMixin
from meerkat.ops.sliceby.sliceby import SliceByCards

Boxable = Union[DataFrame, AbstractColumn, SliceByCards]


class BoxOperation(IdentifiableMixin):

    identifiable_group: str = "box_operations"

    def __init__(
        self,
        fn: Callable,
        args: List[Any],
        kwargs: Dict[str, Any],
        input: Boxable,
        output: Boxable = None,
    ):
        super().__init__()
        if output is None:
            self.output: Boxable = fn(input, *args, **kwargs)
        else:
            self.output = output
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.input = input


class Box(IdentifiableMixin):
    # TODO (all): I think this should probably be a subclassable thing that people
    # implement. e.g. TableInterface

    identifiable_group: str = "boxes"

    def __init__(self, obj: Boxable, id: str = None):
        super().__init__(id)

        self.obj = obj

        self.lineage = [
            BoxOperation(fn=None, args=None, kwargs=None, input=None, output=self.obj)
        ]

    def apply(self, fn: Callable, *args, **kwargs) -> BoxOperation:

        op = BoxOperation(fn=fn, args=args, kwargs=kwargs, input=self.obj)

        self.lineage.append(op)
        self.obj = op.output

        return op

    def undo(self, operation_id: str):
        op_idx = list(map(lambda x: x.id, self.lineage)).index(operation_id)

        self.lineage = self.lineage[:op_idx]
        self.obj = self.lineage[-1].output
        return self.lineage[-1]
