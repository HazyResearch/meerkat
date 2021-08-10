import logging
from typing import Callable

logger = logging.getLogger(__name__)


class LambdaMixin:
    def __init__(self, *args, **kwargs):
        super(LambdaMixin, self).__init__(*args, **kwargs)

    def to_lambda(self, fn: Callable = None):
        from meerkat import LambdaColumn

        return LambdaColumn(data=self, fn=fn)
