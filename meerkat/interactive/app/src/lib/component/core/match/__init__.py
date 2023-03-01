import ast
from dataclasses import dataclass
from typing import Literal

import numpy as np
from fastapi import HTTPException

from meerkat.dataframe import DataFrame
from meerkat.interactive.app.src.lib.component.abstract import Component
from meerkat.interactive.endpoint import Endpoint, EndpointProperty, endpoint
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


def _calc_image_query(df: DataFrame, locs: list, against: str):
    """Calculate the negative samples for a match."""
    return df.loc[locs][against].mean(axis=0)


@endpoint()
def set_criterion(
    df: DataFrame,
    query: str,
    against: str,
    criterion: Store,
    positives: list = None,
    negatives: list = None,
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
        if not query and not negatives and not positives:
            return criterion

        query_embedding = 0.0
        if query:
            query_embedding = parse_query(query)
        if negatives:
            query_embedding = query_embedding - 0.25 * _calc_image_query(
                df, negatives, against
            )
        if positives:
            query_embedding = query_embedding + _calc_image_query(
                df, positives, against
            )

        match_criterion = MatchCriterion(
            query=query,
            against=against,
            query_embedding=query_embedding,
            name=f"match({against}, {query})",
            positives=positives,
            negatives=negatives,
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
    positives: list = None
    negatives: list = None


class OnGetMatchSchemaMatch(EventInterface):
    pass


class OnMatchMatch(EventInterface):
    criterion: MatchCriterion


_get_match_schema = get_match_schema


class Match(Component):
    df: DataFrame
    against: str
    text: str = ""
    encoder: str = "clip"
    title: str = "Match"
    enable_selection: bool = False

    # TODO: Revisit this, how to deal with endpoint interfaces when there is composition
    # and positional arguments
    on_match: EndpointProperty[OnMatchMatch] = None
    get_match_schema: EndpointProperty[OnGetMatchSchemaMatch] = None
    on_clickminus: Endpoint = None
    on_unclickminus: Endpoint = None
    on_clickplus: Endpoint = None
    on_unclickplus: Endpoint = None
    on_reset: Endpoint = None

    def __init__(
        self,
        df: DataFrame = None,
        *,
        against: str,
        text: str = "",
        encoder: str = "clip",
        title: str = "Match",
        enable_selection: bool = False,
        on_match: EndpointProperty = None,
        get_match_schema: EndpointProperty = None,
        on_clickminus: Endpoint = None,
        on_unclickminus: Endpoint = None,
        on_clickplus: Endpoint = None,
        on_unclickplus: Endpoint = None,
        on_reset: Endpoint = None,
    ):
        super().__init__(
            df=df,
            against=against,
            text=text,
            encoder=encoder,
            title=title,
            enable_selection=enable_selection,
            on_match=on_match,
            get_match_schema=get_match_schema,
            on_clickminus=on_clickminus,
            on_unclickminus=on_unclickminus,
            on_clickplus=on_clickplus,
            on_unclickplus=on_unclickplus,
            on_reset=on_reset,
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

        self.negative_selection = Store([], backend_only=True)
        self.positive_selection = Store([], backend_only=True)
        self._mode: Store[
            Literal[
                "set_negative_selection",
                "set_positive_selection",
                "",
            ]
        ] = Store("")

        on_match = set_criterion.partial(
            df=self.df,
            encoder=self.encoder,
            criterion=self._criterion,
            positives=self.positive_selection,
            negatives=self.negative_selection,
        )

        if self.on_match is not None:
            on_match = on_match.compose(self.on_match)

        self.on_match = on_match

    def set_selection(self, selection: Store[list]):
        self.external_selection = selection
        self.enable_selection.set(True)

        self._positive_selection = Store([], backend_only=True)
        self._negative_selection = Store([], backend_only=True)

        self.on_clickminus = self.on_set_negative_selection.partial(self)
        self.on_clickplus = self.on_set_positive_selection.partial(self)

        self.on_unclickminus = self.on_unset_negative_selection.partial(self)
        self.on_unclickplus = self.on_unset_positive_selection.partial(self)

        self.on_reset = self.on_reset_selection.partial(self)

        self.on_external_selection_change(self.external_selection)

    @endpoint()
    def on_reset_selection(self):
        """Reset all the selections."""
        self.negative_selection.set([])
        self.positive_selection.set([])
        self.external_selection.set([])
        self._mode.set("")
        self._positive_selection.set([])
        self._negative_selection.set([])

    @reactive()
    def on_external_selection_change(self, external_selection):
        if self._mode == "set_negative_selection":
            self.negative_selection.set(external_selection)
        elif self._mode == "set_positive_selection":
            self.positive_selection.set(external_selection)

    @endpoint()
    def on_set_negative_selection(self):
        if self._mode == "set_positive_selection":
            self._positive_selection.set(self.external_selection.value)
        self._mode.set("set_negative_selection")
        self.external_selection.set(self._negative_selection.value)

    @endpoint()
    def on_unset_negative_selection(self):
        self._negative_selection.set(self.external_selection.value)
        self._mode.set("")
        self.external_selection.set([])

    @endpoint()
    def on_set_positive_selection(self):
        if self._mode == "set_negative_selection":
            self._negative_selection.set(self.external_selection.value)
        self._mode.set("set_positive_selection")
        self.external_selection.set(self._positive_selection.value)

    @endpoint()
    def on_unset_positive_selection(self):
        self._positive_selection.set(self.external_selection.value)
        self._mode.set("")
        self.external_selection.set([])

    @property
    def criterion(self) -> MatchCriterion:
        return self._criterion
