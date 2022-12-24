from typing import Union, Callable, Mapping, Sequence, TYPE_CHECKING
from inspect import getfullargspec, signature


from meerkat.block.abstract import BlockView
import meerkat.tools.docs as docs

if TYPE_CHECKING:
    from meerkat import DataFrame, Column
    from meerkat.columns.deferred.base import DeferredColumn
    from meerkat.dataframe import DataFrame


_SHARED_DOCS_ = {
    "outputs": docs.ArgDescription(
        """Controls how the output of ``function`` is mapped to the output of the map. 
        Defaults to ``None``.

        *   If ``None``: a single :class:`DeferredColumn` is returned.
        *   If a ``Dict[any, str]``: then a :class:`DataFrame` containing
            DeferredColumns is returned. This is useful when the output of
            ``function`` is a ``Dict``. ``outputs`` maps the outputs of ``function``
            to column names in the resulting :class:`DataFrame`.
        *   If a ``Tuple[str]``: then a :class:`DataFrame` containing 
            output :class:`DeferredColumn` is returned. This is useful when the 
            of ``function`` is a ``Tuple``. ``outputs`` maps the outputs of
            ``function`` to column names in the resulting :class:`DataFrame`.
"""
    )
}


@docs.doc(source=_SHARED_DOCS_, data="data")
def defer(
    data: Union["DataFrame", "Column"],
    function: Callable,
    is_batched_fn: bool = False,
    batch_size: int = 1,
    inputs: Union[Mapping[str, str], Sequence[str]] = None,
    outputs: Union[Mapping[any, str], Sequence[str]] = None,
    output_type: Union[Mapping[str, type], type] = None,
    materialize: bool = True,
) -> Union["DataFrame", "DeferredColumn"]:
    """_summary_

    Learn more in the user guide:  :ref:`guide/dataframe/ops/mapping/deferred`.

    .. note:: 
        This functions is also available as a method of :class:`DataFrame` and
        :class:`Column` under the name ``defer``.

    Examples
    ---------


    Args:
        {data} (DataFrame): The :class:`DataFrame` or :class:`Column` to which the
            function will be applied.
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
            entire dataframe.
        outputs (Union[Dict[any, str], Tuple[str]], optional): {outputs}
        output_type (Union[Dict[str, type], type], optional): _description_. Defaults
            to None.

    Returns:
        Union[DataFrame, DeferredColumn]: A
    """
    from meerkat import DeferredColumn
    from meerkat.block.deferred_block import DeferredBlock, DeferredOp
    from meerkat.columns.abstract import Column
    from meerkat.dataframe import DataFrame

    # prepare arguments for LambdaOp
    if isinstance(data, Column):
        args = [data]
        kwargs = {}
    elif isinstance(data, DataFrame):
        if isinstance(inputs, Mapping):
            args = []
            kwargs = {kw: data[col_name] for col_name, kw in inputs.items()}
        elif isinstance(inputs, Sequence):
            args = [data[col_name] for col_name in inputs]
            kwargs = {}
        elif inputs is None:
            # infer mapping from function signature
            args = []
            kwargs = {}
            for name, param in signature(function).parameters.items():
                if name in data:
                    kwargs[name] = data[name]
                elif param.default is param.empty:
                    raise ValueError(
                        f"Non-default argument {name} does not have a corresponding "
                        f"column in the DataFrame. Please provide an `inputs` mapping "
                        f"pass a lambda function with a different signature."
                    )
        else:
            raise ValueError("`inputs` must be a mapping or sequence.")

    op = DeferredOp(
        fn=function,
        args=args,
        kwargs=kwargs,
        is_batched_fn=is_batched_fn,
        batch_size=batch_size,
        return_format=type(outputs),
    )

    block = DeferredBlock.from_block_data(data=op)

    first_row = op._get(0) if len(op) > 0 else None

    if outputs is None and isinstance(first_row, Mapping):
        # support for splitting a dict into multiple columns without specifying outputs
        outputs = {output_key: output_key for output_key in first_row}
        op.return_format = type(outputs)

    if outputs is None or outputs == "single":
        # can only infer output type if the the input columns are nonempty
        if output_type is None and first_row is not None:
            output_type = type(first_row)

        if not isinstance(output_type, type):
            raise ValueError(
                "Must provide a single `output_type` if `outputs` is None."
            )

        col = DeferredColumn(
            data=BlockView(block_index=None, block=block), output_type=output_type
        )
        return col
    elif isinstance(outputs, Mapping):
        if output_type is None:
            output_type = {
                outputs[output_key]: type(col) for output_key, col in first_row.items()
            }
        if not isinstance(output_type, Mapping):
            raise ValueError(
                "Must provide a `output_type` mapping if `outputs` is a mapping."
            )

        return DataFrame(
            {
                col: DeferredColumn(
                    data=BlockView(block_index=output_key, block=block),
                    output_type=output_type[outputs[output_key]],
                )
                for output_key, col in outputs.items()
            }
        )
    elif isinstance(outputs, Sequence):
        if output_type is None:
            output_type = [type(col) for col in first_row]
        if not isinstance(output_type, Sequence):
            raise ValueError(
                "Must provide a `output_type` sequence if `outputs` is a sequence."
            )
        return DataFrame(
            {
                col: DeferredColumn(
                    data=BlockView(block_index=output_key, block=block),
                    output_type=output_type[output_key],
                )
                for output_key, col in enumerate(outputs)
            }
        )


def map(
    data: Union["DataFrame", "Column"],
    function: Callable,
    is_batched_fn: bool = False,
    batch_size: int = 1,
    inputs: Union[Mapping[str, str], Sequence[str]] = None,
    outputs: Union[Mapping[any, str], Sequence[str]] = None,
    output_type: Union[Mapping[str, type], type] = None,
    materialize: bool = True,
    pbar: bool = False,
    **kwargs,
):
    """
    hkllsdfjklhsdf
    """

    deferred = defer(
        data=data,
        function=function,
        is_batched_fn=is_batched_fn,
        batch_size=batch_size,
        inputs=inputs,
        outputs=outputs,
        output_type=output_type,
        materialize=materialize,
    )
    return _materialize(deferred, batch_size=batch_size, pbar=pbar)


def _materialize(data: Union["DataFrame", "Column"], batch_size: int, pbar: bool):
    from .concat import concat
    from tqdm import tqdm

    result = []
    for batch_start in tqdm(range(0, len(data), batch_size), disable=not pbar):
        result.append(
            data._get(slice(batch_start, batch_start + batch_size, 1), materialize=True)
        )
    return concat(result)
