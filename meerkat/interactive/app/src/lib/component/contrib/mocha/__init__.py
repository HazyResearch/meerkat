import os
from typing import TYPE_CHECKING, List

import numpy as np

from meerkat.interactive.app.src.lib.component.contrib.discover import Discover
from meerkat.interactive.app.src.lib.component.contrib.global_stats import GlobalStats
from meerkat.interactive.app.src.lib.component.contrib.row import Row
from meerkat.interactive.app.src.lib.component.core.filter import FilterCriterion
from meerkat.interactive.app.src.lib.component.deprecate.plot import Plot
from meerkat.ops.sliceby.sliceby import SliceBy
from meerkat.tools.lazy_loader import LazyLoader

from ...abstract import BaseComponent

if TYPE_CHECKING:
    from meerkat.dataframe import DataFrame


manifest = LazyLoader("manifest")
DELTA_COLUMN = "delta"


class ChangeList(BaseComponent):
    code_control: bool = False

    gallery: BaseComponent
    gallery_match: BaseComponent
    gallery_filter: BaseComponent
    gallery_sort: BaseComponent
    gallery_fm_filter: BaseComponent
    gallery_code: BaseComponent
    discover: BaseComponent
    plot: BaseComponent
    active_slice: BaseComponent
    slice_sort: BaseComponent
    slice_match: BaseComponent
    global_stats: BaseComponent
    metric: str = ("Accuracy",)
    v1_name: str = (None,)
    v2_name: str = (None,)

    def __init__(
        self,
        df: "DataFrame",
        v1_column: str,
        v2_column: str,
        label_column: str,
        main_column: str,
        embed_column: str,
        repo_path: str,
        tag_columns: List[str] = None,
        metric: str = "Accuracy",
        v1_name: str = None,
        v2_name: str = None,
        code_control: bool = False,
    ):
        from mocha.repo import SliceRepo

        import meerkat as mk

        v1_name = v1_name or v1_column
        v2_name = v2_name or v2_column

        if tag_columns is None:
            tag_columns = []
        if df.primary_key is None:
            raise ValueError("The DataFrame must have a primary key set.")

        df[DELTA_COLUMN] = df[v2_column] - df[v1_column]

        if os.path.exists(os.path.join(repo_path, "membership.mk")):
            slice_repo = SliceRepo.read(repo_path)
        else:
            slice_repo = SliceRepo(df, repo_path)

        membership_df = slice_repo.membership
        slices_df = slice_repo.slices
        base_examples = df

        with mk.gui.reactive():

            examples_df = mk.merge(base_examples, membership_df, on=df.primary_key_name)

            examples_df = mk.sample(examples_df, len(examples_df))

            # SLICE OVERVIEW
            @mk.gui.reactive
            def compute_slice_scores(examples: mk.DataPanel, slices: mk.DataPanel):
                """Produce a DataFrame with average delta's (and counts) for
                each slice."""

                sb = examples.sliceby(
                    slices["slice_id"].apply(slice_repo._slice_column)
                )
                result = sb[DELTA_COLUMN].mean()

                # compute prototype image embeddings
                image_prototypes_df = sb[[embed_column]].mean(axis=0)
                image_prototypes_df["slice_id"] = image_prototypes_df["slice"].map(
                    slice_repo._slice_id
                )
                image_prototypes_df.remove_column("slice")
                slices = slices.merge(image_prototypes_df, on="slice_id")

                # # Hacky way to rename DELTA_COLUMN column -> count
                count = sb[DELTA_COLUMN].count()
                count["count"] = count[DELTA_COLUMN]
                count = count[["slice", "count"]]
                result = result.merge(count, on="slice")

                result = result.sort(by=DELTA_COLUMN, ascending=False)

                slices["slice"] = slices["slice_id"].apply(slice_repo._slice_column)
                out = (
                    slices[
                        "slice_id",
                        "name",
                        "description",
                        "slice",
                        embed_column,
                    ]
                    .merge(result, on="slice")
                    .drop("slice")
                )
                slices.remove_column("slice")

                return out

            stats_df = compute_slice_scores(examples=examples_df, slices=slices_df)

            @mk.endpoint()
            def append_to_sort(match_criterion, criteria: mk.Store):
                SOURCE = "match"
                criterion = mk.gui.Sort.create_criterion(
                    column=match_criterion.name, ascending=False, source=SOURCE
                )

                criteria.set([criterion] + [c for c in criteria if c.source != SOURCE])

            sort_criteria = mk.Store([])
            match = mk.gui.Match(
                df=base_examples,
                against=embed_column,
                title="Search Examples",
                on_match=append_to_sort.partial(criteria=sort_criteria),
            )
            examples_df, _ = match(examples_df)

            # sort the gallery
            sort = mk.gui.Sort(
                df=examples_df, criteria=sort_criteria, title="Sort Examples"
            )

            # the filter for the gallery
            # TODO(sabri): make this default to the active slice
            filter = mk.gui.Filter(df=examples_df, title="Filter Examples")
            fm_filter = mk.gui.FMFilter(
                df=examples_df,
                manifest_session=manifest.Manifest(
                    client_name="huggingface",
                    client_connection="http://127.0.0.1:7861",
                    temperature=0.1,
                ),
            )
            code = mk.gui.CodeCell()
            current_examples = filter(examples_df)
            current_examples = code(current_examples)
            current_examples = fm_filter(current_examples)
            current_examples = sort(current_examples)

            # removing dirty entries does not use the returned criteria.
            # but we need to pass it as an argument so that the topological sort
            # runs this command after the match.
            slice_sort = mk.gui.Sort(df=stats_df, criteria=[], title="Sort Slices")

            sb_match = mk.gui.Match(
                df=stats_df,
                against=embed_column,
                title="Search Slices",
                on_match=append_to_sort.partial(criteria=slice_sort.criteria),
            )
            stats_df, _ = sb_match()
            stats_df = slice_sort(stats_df)

            @mk.endpoint()
            def on_select_slice(
                slice_id: str,
                criteria: mk.Store,
                code: str,
                query: str,
            ):
                """Update the gallery filter criteria with the selected slice.

                The gallery should be filtered based on the selected
                slice. If no slice is selected, there shouldn't be any
                filtering based on the slice. When the selected slice is
                changed, we should replace the existing filter criteria
                with the new one.
                """
                source = "on_select_slice"
                # wrapped = [
                #     x if isinstance(x, FilterCriterion) else FilterCriterion(**x)
                #     for x in criteria
                # ]
                # on_select_criterion = [x for x in wrapped if x.source == source]
                # assert (
                #     len(on_select_criterion) <= 1
                # ), "Something went wrong - Cannot have more than one selected slice"
                # for x in on_select_criterion:
                #     wrapped.remove(x)
                wrapped = []

                if slice_id:
                    wrapped.append(
                        FilterCriterion(
                            is_enabled=True,
                            column=slice_repo._slice_column(slice_id),
                            op="==",
                            value=True,
                            source=source,
                        )
                    )
                code.set("df")
                query.set("__default__")
                criteria.set(wrapped)
                # set to empty string if None
                # TODO: Need mk.None
                # selected.set(slice_id or "")

            @mk.endpoint()
            def on_remove(slice_id: str, slices_df: mk.DataFrame):
                slice_repo.remove(slice_id)
                mod = mk.gui.DataFrameModification(
                    id=slices_df.id, scope=slices_df.columns
                )
                mod.add_to_queue()
                slice_repo.write()

            @mk.gui.reactive
            def get_selected_slice_id(
                criteria: List[FilterCriterion],
                code: str,
                query: str,
            ):
                if len(criteria) == 1 and query == "__default__" and code == "df":
                    criterion = criteria[0]
                    if criterion.source == "on_select_slice":
                        return slice_repo._slice_id(criterion.column)
                return ""

            selected_slice_id = get_selected_slice_id(
                filter.criteria, code.code, fm_filter.query
            )

            plot = Plot(
                df=stats_df,
                x=DELTA_COLUMN,
                y="name",
                x_label="Accuracy Shift",
                y_label="slice",
                metadata_columns=["count", "description"],
                on_select=on_select_slice.partial(
                    criteria=filter.criteria, query=fm_filter.query, code=code.code
                ),
                on_remove=on_remove.partial(slices_df=slices_df),
            )

            @mk.endpoint()
            def on_write_row(key: str, column: str, value: str, df: mk.DataFrame):
                """Change the value of a column in the slice_df."""
                if not key:
                    return
                df[column][df.primary_key._keyidx_to_posidx(key)] = value
                # We have to force add the dataframe modification to trigger
                # downstream updates
                mod = mk.gui.DataFrameModification(id=df.id, scope=[column])
                mod.add_to_queue()
                slice_repo.write()

            @mk.gui.reactive
            def compute_stats(df: mk.DataFrame):
                return {
                    "count": len(df),
                    f"{metric} Shift": df[DELTA_COLUMN].mean(),
                    f"v1 {metric}": df[v1_column].mean(),
                    f"v2 {metric}": df[v2_column].mean(),
                }

            stats = compute_stats(current_examples)

            @mk.endpoint()
            def on_slice_creation(examples_df: mk.DataFrame):
                current_df = filter(examples_df)
                current_df = code(current_df)
                current_df = fm_filter(current_df)

                slice_id = slice_repo.add(
                    name="Unnamed Slice",
                    membership=mk.DataFrame(
                        {
                            current_df.primary_key_name: current_df.primary_key,
                            "slice": np.ones(len(current_df)),
                        },
                        primary_key=current_df.primary_key_name,
                    ),
                )
                slice_repo.write()
                return slice_id

            active_slice_view = Row(
                df=stats_df,
                selected_key=selected_slice_id,
                columns=["name", "description"],
                stats=stats,
                # rename={""}
                title="Active Slice",
                on_change=on_write_row.partial(
                    df=slices_df
                ),  # the edits should be written on the slices_df
                on_slice_creation=on_slice_creation.partial(
                    examples_df=examples_df
                ).compose(
                    on_select_slice.partial(
                        criteria=filter.criteria,
                        query=fm_filter.query,
                        code=code.code,
                    )
                ),
            )

            @mk.gui.reactive
            def subselect_columns(df):
                return df[
                    list(
                        set(
                            [
                                main_column,
                                DELTA_COLUMN,
                                label_column,
                                v1_column,
                                v2_column,
                                current_examples.primary_key_name,
                            ]
                            + tag_columns
                        )
                    )
                ]

            gallery = mk.gui.Gallery(
                df=subselect_columns(current_examples),
                main_column=main_column,
                tag_columns=tag_columns,
            )

        @mk.endpoint()
        def add_discovered_slices(eb: SliceBy):
            for key, slice in eb.slices.items():
                col = np.zeros(len(slice_repo.membership))
                col[slice] = 1
                # FIXME: make this interface less awful
                slice_repo.add(
                    name=f"discovered_{key}",
                    membership=mk.DataFrame(
                        {
                            eb.data.primary_key_name: eb.data.primary_key,
                            "slice": col,
                        },
                        primary_key=eb.data.primary_key_name,
                    ),
                )

            # add a sort by created time

        discover = Discover(
            df=examples_df,
            by=embed_column,
            target=v1_column,
            pred=v2_column,
            on_discover=add_discovered_slices,
        )

        stats = GlobalStats(
            v1_name=v1_name,
            v2_name=v2_name,
            v1_mean=examples_df[v1_column].mean(),
            v2_mean=examples_df[v2_column].mean(),
            shift=examples_df[DELTA_COLUMN].mean(),
            inconsistency=examples_df[DELTA_COLUMN].std(),
            metric=metric,
        )
        super().__init__(
            gallery_match=match,
            gallery_filter=filter,
            gallery_sort=sort,
            gallery_fm_filter=fm_filter,
            gallery_code=code,
            gallery=gallery,
            discover=discover,
            global_stats=stats,
            active_slice=active_slice_view,
            plot=plot,
            slice_sort=slice_sort,
            slice_match=sb_match,
            v1_name=v1_name,
            v2_name=v2_name,
            metric=metric,
            code_control=code_control,
        )
