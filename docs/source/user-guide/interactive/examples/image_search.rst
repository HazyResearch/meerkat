Building image search
=====================

In this example, we will build an interface to search over images in our data frame.

We will see how we can combine components like :class:`~meerkat.gui.Match`, :class:`~meerkat.gui.Sort`, and :class:`~meerkat.gui.Filter`.

.. collapse:: Code

    .. code-block:: python

        import meerkat as mk

        IMAGE_COLUMN = "img"
        EMBED_COLUMN = "img_clip"


        @mk.endpoint()
        def append_to_sort(match_criterion, criteria: mk.Store):
            """Add match criterion to the sort criteria.

            Args:
                match_criterion: The criterion to add.
                    This is returned by the match component.
                criteria: The current sort criteria.
                    This argument should be bound to a store that will be modified.
            """
            SOURCE = "match"
            criterion = mk.gui.Sort.create_criterion(
                column=match_criterion.name, ascending=False, source=SOURCE
            )
            criteria.set([criterion] + [c for c in criteria if c.source != SOURCE])


        df = mk.get("imagenette", version="160px")
        # Download the precomupted CLIP embeddings for imagenette.
        # You can also embed the images yourself with mk.embed. This will take some time.
        # To embed: df = mk.embed(df, input=IMAGE_COLUMN, out_col=EMBED_COLUMN, encoder="clip").
        df_clip = mk.DataFrame.read(
            "https://huggingface.co/datasets/arjundd/meerkat-dataframes/resolve/main/embeddings/imagenette_160px.mk.tar.gz",  # noqa: E501
            overwrite=True,
        )
        df_clip = df_clip[["img_id", "img_clip"]]
        df = df.merge(df_clip, on="img_id")

        with mk.reactive():
            # Match
            sort_criteria = mk.Store([])
            match = mk.gui.Match(
                df=df,
                against=EMBED_COLUMN,
                title="Search Examples",
                on_match=append_to_sort.partial(criteria=sort_criteria),
            )
            df = match(df)[0]

            # Filter
            filter = mk.gui.Filter(df=df)
            df = filter(df)

            # Sort
            sort = mk.gui.Sort(df=df, criteria=sort_criteria, title="Sort Examples")
            df = sort(df)

            # Gallery
            gallery = mk.gui.Gallery(df=df, main_column="img")

        mk.gui.start(shareable=False)
        mk.gui.Interface(
            component=mk.gui.RowLayout(
                components=[
                    match,
                    mk.gui.RowLayout(components=[filter, sort], classes="grid-cols-2 gap-2"),
                    gallery,
                ],
                classes="grid-cols-1 gap-2",
            )
        ).launch()
