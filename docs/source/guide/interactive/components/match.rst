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
It does not sort the results. However, it can be combined with other operations like :func:`mk.sort` to sort the results as shown below:


.. collapse:: Code

    .. code-block:: python

      IMAGE_COLUMN = "img"
      EMBED_COLUMN = "img_clip"

      path = "~/.meerkat/dataframes/demo/imagenette_clip.mk"
      path = os.path.abspath(os.path.expanduser(path))
      if not os.path.exists(path):
          df = mk.get("imagenette", version="160px")

          # Embed the image.
          # This can take a while on the CPU.
          df: mk.DataFrame = mk.embed(df, input=IMAGE_COLUMN, out_col=EMBED_COLUMN)
          df.write("~/.meerkat/dataframes/imagenette_clip.mk")
      else:
          df = mk.DataFrame.read(path)

      with mk.gui.react():
          # Match
          match = mk.gui.Match(df=df, against=EMBED_COLUMN)
          examples_df = match(df)[0]

          # Sort - Takes the output of match and sorts it.
          df_sorted = mk.sort(data=examples_df, by=match.criterion.name, ascending=False)

      # Gallery - To display the results
      gallery = mk.gui.Gallery(
          df=df_sorted,
          main_column=IMAGE_COLUMN,
      )

      mk.gui.start(shareable=False)
      mk.gui.Interface(component=mk.gui.RowLayout(components=[match, gallery])).launch()


.. raw:: html

  <div style="position: relative; padding-bottom: 62.14039125431531%; height: 0;"><iframe src="https://www.loom.com/embed/cda949c144054320ac2ede5dee76f460" frameborder="0" webkitallowfullscreen mozallowfullscreen allowfullscreen style="position: absolute; top: 0; left: 0; width: 100%; height: 100%;"></iframe></div>

