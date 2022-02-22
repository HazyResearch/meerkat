import logging
from typing import Callable, Dict, Optional, Tuple, Union
from meerkat.block.abstract import BlockView

from meerkat.block.lambda_block import LambdaOp, LambdaBlock


logger = logging.getLogger(__name__)


class LambdaMixin:
    def __init__(self, *args, **kwargs):
        super(LambdaMixin, self).__init__(*args, **kwargs)

    def to_lambda(
        self, 
        function: Callable, 
        is_batched_fn: bool = False,
        batch_size: int = 1, 
        outputs: Union[type, Tuple[str], Dict[str, type]] = None
    ):
        from meerkat.columns.abstract import AbstractColumn
        from meerkat.datapanel import DataPanel
        from meerkat import LambdaColumn
        
        if isinstance(self, AbstractColumn):
            inputs = {0: self}
        elif isinstance(self, DataPanel):
            inputs = {

            }
        
        op = LambdaOp(
            fn=function, 
            inputs=inputs,
            is_batched_fn=is_batched_fn,
            batch_size=batch_size,
        )

        block = LambdaBlock.from_block_data(data=op)


        if (outputs is None) or isinstance(outputs, type):
            col = LambdaColumn(
                data=BlockView(block_index=None, block=block),
                output_type=outputs
            )
            return col


        return DataPanel(
            {
                col: BlockView(block_index=output_key, block=block)

                for output_key in outputs 
            }
        )
