import ast
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from fastapi import HTTPException

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import EndpointProperty, endpoint
from meerkat.interactive.event import EventInterface
from meerkat.interactive.graph import Store, reactive

_SUPPORTED_BIN_OPS = {
    "Add": lambda x, y: x + y,
    "Sub": lambda x, y: x - y,
    "Mult": lambda x, y: x * y,
    "Div": lambda x, y: x / y,
    "Pow": lambda x, y: x**y,
}

_SUPPORTED_CALLS = {
    "concat": lambda *args: np.concatenate(args, axis=1),
}


@endpoint()
def refine_match(
    df: DataFrame, 
    against: str,
    match_column: str,
    label_column: str,
):
    """
    Refine a match, using feedback from the user.

    Labels are +1 for a match, -1 for a non-match, and 0 for unlabeled.
    """
    # Train a logistic regression model to predict the label.
    from sklearn.linear_model import LogisticRegression

    # Mask for positive and negative labels.
    mask_p = df[label_column] == 1
    mask_n = df[label_column] == -1

    # Embeddings
    X = df[against][mask_p | mask_n]
    y = df[label_column][mask_p | mask_n]

    # Train a logistic regression model to predict the label.
    model = LogisticRegression()
    model.fit(X, y)

    # Predict the label for all rows.
    y_pred = model.predict(df[against])

    # Update the match column.
    df[f'{match_column}_refined'] = y_pred
    df.set(df)



def parse_query(query: str):
    return _parse_query(ast.parse(query, mode="eval").body)


def _parse_query(
    node: ast.AST,
):
    import meerkat as mk

    if isinstance(node, ast.BinOp):
        return _SUPPORTED_BIN_OPS[node.op.__class__.__name__](
            _parse_query(node.left), _parse_query(node.right)
        )
    elif isinstance(node, ast.Call):
        return _SUPPORTED_CALLS[node.func.id](*[_parse_query(arg) for arg in node.args])
    elif isinstance(node, ast.Constant):
        return mk.embed(
            data=mk.column([node.value]),
            encoder="clip",
            num_workers=0,
            pbar=False,
        )
    else:
        node_repr = node.id if hasattr(node, "id") else node
        if isinstance(node_repr, str):
            node_repr = f"'{node_repr}'"
        raise ValueError(f"Unsupported query {node_repr}")


@endpoint()
def get_match_schema(df: DataFrame):
    import meerkat as mk
    from meerkat.interactive.api.routers.dataframe import (
        SchemaResponse,
        _get_column_infos,
    )

    columns = [
        k
        for k, v in df.items()
        if isinstance(v, mk.TensorColumn) and len(v.shape) == 2
        # TODO: We should know the provenance of embeddings and where they came from,
        # to explicitly check whether the encoder will match it in size.
    ]
    return SchemaResponse(
        id=df.id,
        columns=_get_column_infos(df, columns),
        nrows=len(df),
    )


@endpoint()
def set_criterion(
    df: DataFrame,
    query: str,
    against: str,
    criterion: Store,
    encoder: str = None,
):
    """Match a query string against a DataFrame column.

    The `dataframe_id` remains the same as the original request.
    """
    if not isinstance(df, DataFrame):
        raise HTTPException(
            status_code=400, detail="`match` expects a ref containing a dataframe"
        )

    try:
        query_embedding = parse_query(query)

        match_criterion = MatchCriterion(
            query=query,
            against=against,
            query_embedding=query_embedding,
            name=f"match({against}, {query})",
        )
        criterion.set(match_criterion)

        if not (criterion.value is None or criterion.against is None):
            data_embedding = df[criterion.against]
            scores = (data_embedding @ criterion.query_embedding.T).squeeze()
            df[criterion.name] = scores
            df.set(df)

    except Exception as e:
        raise e

    return criterion


@dataclass
class MatchCriterion:
    against: str
    query: str
    name: str
    query_embedding: np.ndarray = None


class OnGetMatchSchemaMatch(EventInterface):
    pass


class OnMatchMatch(EventInterface):
    criterion: MatchCriterion


@reactive(nested_return=True)
def compute_match_scores(
    df: DataFrame,
    criterion: MatchCriterion,
) -> Tuple[DataFrame, str]:
    """Compute the match scores for a given criterion.

    Args:
        df (DataFrame): The DataFrame to match against.
        criterion (MatchCriterion): The criterion to match against.

    Returns:
        Tuple[DataFrame, str]: The DataFrame with the match scores
            added as a column, and the name of the column.
    """
    df = df.view()
    if criterion == None or criterion.against is None:  # noqa: E711
        return df, None

    data_embedding = df[criterion.against]
    scores = (data_embedding @ criterion.query_embedding.T).squeeze()
    df[criterion.name] = scores

    return df, criterion.name


_get_match_schema = get_match_schema


class Match(Component):
    df: DataFrame
    against: str
    text: str = ""
    encoder: str = "clip"
    title: str = "Match"

    # TODO: Revisit this, how to deal with endpoint interfaces when there is composition
    # and positional arguments
    on_match: EndpointProperty[OnMatchMatch] = None
    get_match_schema: EndpointProperty[OnGetMatchSchemaMatch] = None

    def __init__(
        self,
        df: DataFrame = None,
        *,
        against: str,
        text: str = "",
        encoder: str = "clip",
        title: str = "Match",
        on_match: EndpointProperty = None,
        get_match_schema: EndpointProperty = None,
    ):
        super().__init__(
            df=df,
            against=against,
            text=text,
            encoder=encoder,
            title=title,
            on_match=on_match,
            get_match_schema=get_match_schema,
        )

        # we do not add the against or the query to the partial, because we don't
        # want them to be maintained on the backend
        # if they are maintained on the backend, then a store update dispatch will
        # run on every key stroke
        self.get_match_schema = _get_match_schema.partial(df=self.df)

        self._criterion: MatchCriterion = Store(
            MatchCriterion(against=None, query=None, name=None),
            backend_only=True,
        )

        on_match = set_criterion.partial(
            df=self.df,
            encoder=self.encoder,
            criterion=self._criterion,
        )
        if self.on_match is not None:
            on_match = on_match.compose(self.on_match)

        self.on_match = on_match

    @property
    def criterion(self) -> MatchCriterion:
        return self._criterion

    def __call__(self, df: DataFrame = None) -> DataFrame:
        if df is None:
            df = self.df

        return compute_match_scores(df, self.criterion)
