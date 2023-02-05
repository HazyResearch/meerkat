from typing import Any, Dict, List, Optional, Sequence, Union

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint, EndpointProperty, endpoint
from meerkat.interactive.graph import Store, react


@react()
def run_code_cell(df: DataFrame, code: str):
    df = df.view()  # this is needed to avoid cycles in simple df case
    lines = code.split("\n")
    _locals = locals()
    exec("\n".join(lines[:-1]), {}, _locals)
    return eval(lines[-1], {}, _locals)


@endpoint
def base_on_run(code: Store[str], new_code: str):
    # TODO: there is some checks we can do here, before setting (e.g. empty string, syntax checks)
    code.set(new_code)
    return code


class CodeCell(Component):
    code: str = ""
    on_run: EndpointProperty = None

    def __init__(self, code: str = "df", on_run: Endpoint= None):
        code = Store(code)
        if on_run is not None:
            on_run = base_on_run.partial(code=code).compose(on_run)
        else:
            on_run = base_on_run.partial(code=code)

        super().__init__(on_run=on_run, code=code)


    def __call__(self, df: DataFrame):
        out = run_code_cell(df, self.code)
        if not isinstance(out, DataFrame):
            raise ValueError("The code must return a DataFrame.")
        return out
