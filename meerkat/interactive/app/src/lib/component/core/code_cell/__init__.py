from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint, EndpointProperty, endpoint
from meerkat.interactive.event import EventInterface
from meerkat.interactive.graph import Store, reactive
from meerkat.interactive.graph.marking import mark


@reactive()
def run_code_cell(df: DataFrame, code: str):
    df = df.view()  # this is needed to avoid cycles in simple df case
    lines = code.split("\n")
    _locals = locals()
    exec("\n".join(lines[:-1]), {}, _locals)
    return eval(lines[-1], {}, _locals)


@reactive()
def run_filter_code_cell(df: DataFrame, code: str):
    df = df.view()  # this is needed to avoid cycles in simple df case
    _locals = locals()
    exec(code, {}, _locals)
    return eval("df[df.map(condition)]", {}, _locals)


@endpoint()
def base_on_run(code: Store[str], new_code: str):
    # TODO: there is some checks we can do here,
    # before setting (e.g. empty string, syntax checks)
    code.set(new_code)
    return code


class OnRunCodeCell(EventInterface):
    new_code: str


class CodeCell(Component):
    code: str = ""
    on_run: EndpointProperty[OnRunCodeCell] = None

    def __init__(self, code: str = "df", on_run: Endpoint = None):
        code = mark(code)
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


class FilterCodeCell(CodeCell):
    def __init__(self, df: DataFrame, code: str = None, on_run: Endpoint = None):
        code = f"def condition({', '.join(map(lambda x: x.value, df.columns[:4]))})"
        ":\n   return True"
        super().__init__(code=code, on_run=on_run)

    def __call__(self, df: DataFrame):
        out = run_filter_code_cell(df, self.code)
        if not isinstance(out, DataFrame):
            raise ValueError("The code must return a DataFrame.")
        return out
