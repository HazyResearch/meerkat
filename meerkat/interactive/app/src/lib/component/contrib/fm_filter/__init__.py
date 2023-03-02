import hashlib
from typing import Sequence

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint, EndpointProperty, endpoint
from meerkat.interactive.graph import Store, reactive, unmarked


@endpoint
def base_on_run(
    new_query: str,
    query: str,
    df: DataFrame,
    criteria_df: DataFrame,
    manifest_session,
):
    print(new_query)
    if new_query == "" or new_query == "__default__":
        query.set("__default__")
        return

    variables = {}
    exec(new_query, None, variables)
    prompt, answers = variables["prompt"], variables["answers"]

    df = df.view()

    def _run_inference(question: Sequence[str]):
        return manifest_session.run(
            [prompt.format(question=question) for question in question]
        )

    # df = df.sample(100)  # TODO: remove this
    # create a column name out of a hash of the query
    print("endpoint", new_query)
    column_name = _hash_query(new_query)
    if column_name not in criteria_df:
        out = df.map(_run_inference, is_batched_fn=True, batch_size=200, pbar=True)

        criteria_df[column_name] = criteria_df.primary_key.isin(
            df[out.isin(answers)].primary_key
        )

    query.set(new_query)


@reactive
def filter_df(df: DataFrame, criteria_df: DataFrame, query: str):
    df = df[
        df.primary_key.isin(criteria_df.primary_key[criteria_df[_hash_query(query)]])
    ]
    return df


class FMFilter(Component):
    query: str
    criteria_df: DataFrame
    on_run: EndpointProperty = None

    def __init__(self, df=DataFrame, on_run: Endpoint = None, manifest_session=None):
        query = Store("__default__")
        with unmarked():
            criteria_df = df[[df.primary_key_name]]
            criteria_df[_hash_query("__default__")] = True
        manifest_session = manifest_session

        partial_on_run = base_on_run.partial(
            df=df,
            query=query,
            criteria_df=criteria_df,
            manifest_session=manifest_session,
        )

        if on_run is not None:
            on_run = partial_on_run.compose(on_run)
        else:
            on_run = partial_on_run

        super().__init__(query=query, criteria_df=criteria_df, on_run=on_run)

    def __call__(self, df: DataFrame):
        out = filter_df(df, criteria_df=self.criteria_df, query=self.query)
        return out


def _hash_query(query: str):
    return "fm_" + hashlib.sha256(query.encode()).hexdigest()[:8]
