import logging
from typing import TYPE_CHECKING, Callable, Mapping, Sequence, Type, Union

from pandas.util._decorators import doc

from meerkat.ops.map import defer

if TYPE_CHECKING:
    from meerkat.columns.abstract import Column
    from meerkat.columns.deferred.base import DeferredColumn
    from meerkat.dataframe import DataFrame

logger = logging.getLogger(__name__)


class DeferrableMixin:
    def __init__(self, *args, **kwargs):
        super(DeferrableMixin, self).__init__(*args, **kwargs)

    @doc(defer, data="self")
    def defer(
        self,
        function: Callable,
        is_batched_fn: bool = False,
        batch_size: int = 1,
        inputs: Union[Mapping[str, str], Sequence[str]] = None,
        outputs: Union[Mapping[any, str], Sequence[str]] = None,
        output_type: Union[Mapping[str, Type["Column"]], Type["Column"]] = None,
        materialize: bool = True,
    ) -> Union["DataFrame", "DeferredColumn"]:
        return defer(
            data=self,
            function=function,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            inputs=inputs,
            outputs=outputs,
            output_type=output_type,
            materialize=materialize,
        )
