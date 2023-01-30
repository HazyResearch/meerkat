# from typing import Any, Dict, List, Optional, Sequence, Union

# import numpy as np
# import pandas as pd
# from pydantic import BaseModel, Field

# from meerkat.columns.abstract import Column
# from meerkat.columns.scalar import ScalarColumn
# from meerkat.dataframe import DataFrame
# from meerkat.interactive.endpoint import Endpoint, endpoint
# from meerkat.interactive.graph import Store, react

# from ..abstract import BaseComponent


# @react()
# def run_formula_bar(df: DataFrame, code: str):
#     df = df.view()  # this is needed to avoid cycles in simple df case
#     lines = code.split("\n")
#     _locals = locals()
#     exec("\n".join(lines[:-1]), {}, _locals)
#     return eval(lines[-1], {}, _locals)


# @endpoint
# def on_run(code: Store[str], new_code: str):
#     print("is endpoint running")
#     code.set(new_code)
#     return code


# class FormulaBar(BaseComponent):
#     on_run: Endpoint = None

#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.code = Store("df", backend_only=True)
#         self.on_run = on_run.partial(code=self.code)

#     def __call__(self, df: DataFrame):
#         out = run_formula_bar(df, self.code)
#         if not isinstance(out, DataFrame):
#             raise ValueError("The code must return a DataFrame.")
#         return out
