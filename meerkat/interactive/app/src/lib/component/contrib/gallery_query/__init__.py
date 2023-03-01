from typing import TYPE_CHECKING

from meerkat.interactive.app.src.lib.component.core.carousel import Carousel
from meerkat.interactive.app.src.lib.component.core.gallery import Gallery
from meerkat.interactive.app.src.lib.component.core.markdown import Caption
from meerkat.interactive.app.src.lib.component.core.match import Match
from meerkat.interactive.graph.magic import magic
from meerkat.interactive.graph.reactivity import reactive
from meerkat.ops.cond import _bool as mkbool
from meerkat.ops.cond import cor

from ...html import div, flex, flexcol

if TYPE_CHECKING:
    from meerkat import DataFrame


class GalleryQuery(div):
    # match: Match = None
    # gallery: Gallery = None

    def __init__(
        self,
        df: "DataFrame",
        main_column: str,
        against: str,
        classes: str = "h-screen",
    ):
        from meerkat.ops.sort import sort

        match = Match(
            df=df,
            against=against,
        )

        df = df.mark()

        # Get the name of the match criterion in a reactive way.
        with magic():
            criterion_name = match.criterion.name

        # Sort
        df_sorted = sort(data=df, by=criterion_name, ascending=False)

        # Gallery
        with magic():
            allow_selection = cor(False, mkbool(match._mode))

        gallery = Gallery(
            df_sorted,
            main_column=main_column,
            allow_selection=allow_selection,
        )
        match.set_selection(gallery.selected)

        @reactive()
        def get_positives(df, positive_selection, _positive_selection, selection, mode):
            if mode == "set_positive_selection":
                selected = positive_selection
            else:
                selected = _positive_selection
            if selected:
                return df.loc[selected]
            else:
                return df.head(0)

        @reactive()
        def get_negatives(df, negative_selection, _negative_selection, selection, mode):
            if mode == "set_negative_selection":
                selected = negative_selection
            else:
                selected = _negative_selection
            if selected:
                return df.loc[selected]
            else:
                return df.head(0)

        positive_caption_classes = reactive(
            lambda mode: "text-sm self-center mb-1"
            if mode == "set_positive_selection"
            else "text-sm self-center mb-1 text-gray-400"
        )
        negative_caption_classes = reactive(
            lambda mode: "text-sm self-center mb-1"
            if mode == "set_negative_selection"
            else "text-sm self-center mb-1 text-gray-400"
        )

        carousel_positives = Carousel(
            get_positives(
                df,
                match.positive_selection,
                match._positive_selection,
                gallery.selected,
                match._mode,
            ),
            main_column=main_column,
        )
        carousel_negatives = Carousel(
            get_negatives(
                df,
                match.negative_selection,
                match._negative_selection,
                gallery.selected,
                match._mode,
            ),
            main_column=main_column,
        )

        df.unmark()

        component = div(
            [
                match,
                flex(
                    [
                        flexcol(
                            [
                                Caption(
                                    "Negative Query Images Selected",
                                    classes=negative_caption_classes(match._mode),
                                ),
                                carousel_negatives,
                            ],
                            classes="flex-1 justify-center align-middle",
                        ),
                        flexcol(
                            [
                                Caption(
                                    "Positive Query Images Selected",
                                    classes=positive_caption_classes(match._mode),
                                ),
                                carousel_positives,
                            ],
                            classes="flex-1 justify-center align-middle",
                        ),
                    ],
                    classes="justify-center gap-4 my-2",
                ),
                gallery,
            ],
            classes="h-full grid grid-rows-[auto,auto,4fr]",
        )
        super().__init__(
            slots=[component],
            classes=classes,
        )
        self.match = match
        self.gallery = gallery

    def _get_ipython_height(self) -> str:
        return "1000px"
