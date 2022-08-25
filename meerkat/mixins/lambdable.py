import logging
from typing import TYPE_CHECKING, Callable, Mapping, Sequence, Union

from pandas.util._decorators import doc

from meerkat.block.abstract import BlockView

if TYPE_CHECKING:
    from meerkat.columns.abstract import AbstractColumn
    from meerkat.columns.lambda_column import LambdaColumn
    from meerkat.datapanel import DataPanel

logger = logging.getLogger(__name__)


@doc(data="data")
def to_lambda(
    data: Union["DataPanel", "AbstractColumn"],
    function: Callable,
    is_batched_fn: bool = False,
    batch_size: int = 1,
    inputs: Union[Mapping[str, str], Sequence[str]] = None,
    outputs: Union[Mapping[any, str], Sequence[str]] = None,
    output_type: Union[Mapping[str, type], type] = None,
) -> Union["DataPanel", "LambdaColumn"]:
    """_summary_

    Examples
    ---------


    Args:
        {data}:
        function (Callable): The function that will be applied to the rows of
            ``{data}``.
        is_batched_fn (bool, optional): Whether the function must be applied on a
            batch of rows. Defaults to False.
        batch_size (int, optional): The minimum batch size . Ignored if
            ``is_batched_fn=False``.  Defaults to 1.
        inputs (Dict[str, str], optional): Dictionary mapping column names in
            ``{data}`` to keyword arguments of ``function``. Ignored if ``{data}`` is a
            column. When calling ``function`` values from the columns will be fed to
            the corresponding keyword arguments. Defaults to None, in which case the
            entire datapanel.
        outputs (Union[Dict[any, str], Tuple[str]], optional): Controls how the
            output of ``function`` is mapped to the returned
            :class:`LambdaColumn`(s). Defaults to None.
            * If ``None``, a single :class:`LambdaColumn` is returned.
            * If a ``Dict[any, str]``, then a :class:`DataPanel` containing
            :class:`LambdaColumn`s is returned. This is useful when the output of
            ``function`` is a ``Dict``. ``outputs`` maps the outputs of ``function``
            to column names in the resulting :class:`DataPanel`.
            * If a ``Tuple[str]``, then a :class:`DataPanel` containing
            :class:`LambdaColumn`s is returned. , This is useful when the output of
            ``function`` is a ``Tuple``. ``outputs`` maps the outputs of
            ``function`` to column names in the resulting :class:`DataPanel`.
        output_type (Union[Dict[str, type], type], optional): _description_. Defaults
            to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    from meerkat import LambdaColumn
    from meerkat.block.lambda_block import LambdaBlock, LambdaOp
    from meerkat.columns.abstract import AbstractColumn
    from meerkat.datapanel import DataPanel

    # prepare arguments for LambdaOp
    if isinstance(data, AbstractColumn):
        args = [data]
        kwargs = {}
    elif isinstance(data, DataPanel):
        if isinstance(inputs, Mapping):
            args = []
            kwargs = {kw: data[col_name] for col_name, kw in inputs.items()}
        elif isinstance(inputs, Sequence):
            args = [data[col_name] for col_name in inputs]
            kwargs = {}
        elif inputs is None:
            args = [data]
            kwargs = {}
        else:
            raise ValueError("")

    op = LambdaOp(
        fn=function,
        args=args,
        kwargs=kwargs,
        is_batched_fn=is_batched_fn,
        batch_size=batch_size,
        return_format=type(outputs),
    )

    block = LambdaBlock.from_block_data(data=op)

    if outputs is None:
        if not (isinstance(output_type, type) or output_type is None):
            raise ValueError

        col = LambdaColumn(
            data=BlockView(block_index=None, block=block), output_type=output_type
        )
        return col
    elif isinstance(outputs, Mapping):
        return DataPanel(
            {
                col: LambdaColumn(
                    data=BlockView(block_index=output_key, block=block),
                    output_type=output_type[output_key],
                )
                for output_key, col in outputs.items()
            }
        )
    elif isinstance(outputs, Sequence):
        return DataPanel(
            {
                col: LambdaColumn(data=BlockView(block_index=output_key, block=block))
                for output_key, col in enumerate(outputs)
            }
        )


class LambdaMixin:
    def __init__(self, *args, **kwargs):
        super(LambdaMixin, self).__init__(*args, **kwargs)

    @doc(to_lambda, data="self")
    def to_lambda(
        self,
        function: Callable,
        is_batched_fn: bool = False,
        batch_size: int = 1,
        inputs: Union[Mapping[str, str], Sequence[str]] = None,
        outputs: Union[Mapping[any, str], Sequence[str]] = None,
        output_type: Union[Mapping[str, type], type] = None,
    ) -> Union["DataPanel", "LambdaColumn"]:
        return to_lambda(
            data=self,
            function=function,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
            inputs=inputs,
            outputs=outputs,
            output_type=output_type,
        )
