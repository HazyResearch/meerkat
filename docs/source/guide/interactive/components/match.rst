:class:`mk.gui.Match`
=====================


What is matching?
-----------------

We are often interested in determining how similar two data points are.
These data points can be of the same or of different modalities (i.e. image, text, audio, etc.).
For example, we may be interested in seeing how similar the word "red" is to a picture of a car to determine if it is a red car.

One way of doing this is computing the inner product between the embedding of the text "red" (i.e. *query*) and the embedding of the image of the red car (i.e. *target*).
We can also scale this to compute the inner product between the query and multiple targets. This process is known as *matching*.

The component
-------------
The :class:`mk.gui.Match` component renders an interface for matching.

The user can type in the query they would like to run in the text box.
They can also select the column corresponding to the target embedding in the dropdown.

Note: :class:`mk.gui.Match` 1) embeds the text query and 2) computes the inner product between the query embedding and the target embedding(s).
It does not sort the results. However, it can be combined with other operations like :func:`mk.sort` to sort the results.

Example
-------
Let's look at creating an image search interface for the imagenette dataset.
We can combine :class:`mk.gui.Match` with other operations (:func:`mk.sort`) and components (:class:`mk.gui.Gallery`) to search over images in the dataset.

You can also do arithmetic on the embeddings. For example, you can subtract similarity scores between queries (e.g. ``"dog" - "cat"``)

.. collapse:: Code

    .. code-block:: python

        import meerkat as mk

        IMAGE_COLUMN = "img"
        EMBED_COLUMN = "img_clip"

        df = mk.get("imagenette", version="160px")
        # Download the precomupted CLIP embeddings for imagenette.
        # You can also embed the images yourself with mk.embed. This will take some time.
        # To embed: df = mk.embed(df, input=IMAGE_COLUMN, out_col=EMBED_COLUMN, encoder="clip").
        df_clip = mk.DataFrame.read("https://huggingface.co/datasets/arjundd/meerkat-dataframes/resolve/main/imagenette_clip.mk.tar.gz")
        df = df.merge(df_clip, on="img_id")

        with mk.gui.react():
            # Match
            match = mk.gui.Match(df=df, against=EMBED_COLUMN)
            examples_df = match(df)[0]

            # Sort
            df_sorted = mk.sort(data=examples_df, by=match.criterion.name, ascending=False)

        # Gallery
        gallery = mk.gui.Gallery(
            df=df_sorted,
            main_column=IMAGE_COLUMN,
        )

        mk.gui.start(shareable=False)
        mk.gui.Interface(component=mk.gui.RowLayout(components=[match, gallery])).launch()

.. raw:: html

    <div style="position: relative; padding-bottom: 62.14039125431531%; height: 0;"><iframe src="https://www.loom.com/embed/bc1d4e145b6946b4ac8f25e721685bb5" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>
