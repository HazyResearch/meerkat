Quickstart
========================

This page gives a quick overview on how to start using Robustness Gym.

The central operation in Robustness Gym is the construction of *slices* of data:
a slice is just a dataset that is used to test specific model properties.

Robustness Gym comes with a set of general abstractions to build slices with ease.
We'll use a simple example to show you how these work.

Robustness Gym also has a lot of built-in functionality that you can use out-of-the-box 
(thanks to some other great open-source projects) for creating slices. You can
read more about these in [](), and check out []() if you'd like to
contribute some of your own slice building code to Robustness Gym.

Let's dive in quickly!

Building Slices
---------------------  

Robustness Gym contains a ``SliceBuilder`` class for writing code to build slices.
This
class defines a common interface that all ``SliceBuilders`` must follow:

#.  Any ``SliceBuilder`` object can be called using ``slicebuilder(batch_or_dataset,
    columns)``.

#.  This call always returns a ``(dataset, slices, matrix)`` tuple.

To see how this works, let's see a simple example. We're going to

#.  Create a dummy dataset containing just 4 text examples.

#.  Use a ``ScoreSubpopulation`` (a kind of ``SliceBuilder``) to build 2 slices.

Let's start by creating the dataset.

.. code-block:: python

    from robustnessgym import Dataset, Identifier

    dataset = Dataset.from_batch({
        'text': ['a person is walking',
                 'a person is running',
                 'a person is sitting',
                 'a person is walking on a street eating a bagel']
    }, identifier=Identifier(_name='MyDataset'))


Here, we used the ``.from_batch(..)`` method to create a dataset called ``MyDataset``.
This dataset has a single column called `text` with 4 examples or rows.

The ``Identifier`` class is used to store identifying information for ``Dataset``
objects, ``SliceBuilder`` objects and more.

.. tip::
    Most objects in Robustness Gym have a ``.identifier`` property that can be used to
    inspect the object.

Next, let's create the ``ScoreSubpopulation`` to build slices.

.. code-block:: python

    def length(batch, columns):
        """
        A simple function to compute the length of all examples in a batch.

        batch: a dict of lists
        columns: a list of str

        return: a list of lengths
        """
        assert len(columns) == 1, "Pass in a single column."

        # The name of the column to grab text from
        column_name = columns[0]
        text_batch = batch[column_name]

        # Tokenize the text using .split() and calculate the number of tokens
        return [len(text.split()) for text in text_batch]


We pause here to point out three things:

#.  The ``def func(batch, columns)`` is a common pattern in Robustness Gym for
    adding custom functionality.

    The ``batch`` here refers to a batch of data,

    .. code-block:: python

        {'text': ['a person is walking', 'a person is running'], 'index': [0, 1]}

    is a batch of size 2 from the dataset (``dataset[:2]``).

    The ``columns`` parameter specifies the relevant columns of the batch.
    This has some advantages e.g. suppose ``otherdataset`` has a column of text named
    `sentence` instead.
    We can reuse ``length`` for both datasets,

    .. code-block:: python

        length(batch=dataset[:2], columns=['text'])
        length(batch=otherdataset[:2], columns=['sentence'])

#.  ``length`` returns a list of scores (lengths in this case). This is an
    important ingredient of the ``ScoreSubpopulation``, which constructs (as the
    name suggests) slices by bucketing examples based on their score.

#.  We tokenized text inside the length function. This is bad:

    #.  Tokenization is a basic step in text processing, and we should only do it once.
    #.  If it was some other, more expensive operation, we should definitely do it once.

Let's keep going and wrap ``length`` in a ``ScoreSubpopulation``.


.. code-block:: python

    from robustnessgym import ScoreSubpopulation

    # Create the score subpopulation for length
    length_sp = ScoreSubpopulation(intervals=[(0, 5), (5, 10)], score_fn=length)


The ``ScoreSubpopulation`` requires

#.  a list of ``intervals``, each interval is a tuple containing the range of lengths
    that are considered part of that slice.
#.  a ``score_fn``, used to assign scores to a batch of examples

Let's run this on the dataset.

.. code-block:: python

    # Run the length subpopulation on the dataset
    dataset, slices, membership = length_sp(batch_or_dataset=dataset, columns=['text'])


This call just executes the ``length`` function on the dataset, and buckets the
examples based on which intervals they fall in. As we briefly mentioned earlier, this
returns the ``(dataset, slices, membership)`` tuple,

#.  ``dataset`` now tags each example with slice information i.e. what slices does
    the example belong to
#.  ``slices`` is a list of ``Slice`` objects (2 here, since we specified 2
    intervals). Each ``Slice`` object is a dataset containing just the examples that
    were part of the slice.
#.  ``membership`` is a ``np.array`` matrix of shape ``(n, m)``, where ``n`` is the
    number of examples in the original dataset, and ``m`` is the number of slices
    built. Entry ``(i, j)`` is 1 if example ``i`` is in slice ``j``.

And that's (almost) it! Most code you write in Robustness Gym will follow a
similar workflow. Before we end, we take a short segue to talk about the other major
abstraction in Robustness Gym: the ``CachedOperation`` class.

Caching Information
---------------------

As we noted earlier, we tokenized text inside the ``length`` function, when we should
ideally run this step separately and reuse it across multiple ``SliceBuilder`` objects.

When creating Robustness Gym, we noticed this pattern frequently: cache
some information (``CachedOperation``), and use that information to build some slices
(``SliceBuilder``).

Let's look at the same example as before, and use a ``CachedOperation`` for
tokenization this time.


.. code-block:: python

    from robustnessgym import CachedOperation, Identifier

    def tokenize(batch, columns):
        """
        A simple function to tokenize a batch of examples.

        batch: a dict of lists
        columns: a list of str

        return: a list of tokenized text
        """
        assert len(columns) == 1, "Pass in a single column."

        # The name of the column to grab text from
        column_name = columns[0]
        text_batch = batch[column_name]

        # Tokenize the text using .split()
        return [text.split() for text in text_batch]

    # Create the CachedOperation
    cachedop = CachedOperation(apply_fn=tokenize,
                               identifier=Identifier(_name="Tokenizer"))


We've written ``tokenize`` with the familiar ``func(batch, columns)`` function
signature. This function is then wrapped into a ``CachedOperation`` for use.

.. tip::
    A ``CachedOperation`` can be created with *any* ``func(batch, columns)``. The only
    constraint is that it must return a list, with size equal to that of the batch.


Let's create our ``ScoreSubpopulation`` for length again.

.. code-block:: python

    from robustnessgym.decorators import singlecolumn

    def length(batch, columns):
        """
        A simple function to compute the length of all examples in a batch.

        batch: a dict of lists
        columns: a list of str

        return: a list of lengths
        """
        assert len(columns) == 1, "Pass in a single column."

        # The name of the column to grab text from
        column_name = columns[0]
        text_batch = batch[column_name]

        CachedOperation.retrieve(
            batch=batch,
            columns=[column_name],
            proc_fns=lambda decoded_batch: []
        )

        # Tokenize the text using .split() and calculate the number of tokens
        return [len(text.split()) for text in text_batch]



Robustness Gym ships with ``CachedOperations`` that use standard text processing
pipelines to tokenize and tag text.


There's a ton more to Robustness Gym (and more coming).
Here are some pointers on where to head to next, depending on your specific goals:

#.  If you want a more detailed tutorial and walkthrough, head to the [Tutorial 1]()
    Jupyter notebook
#.  If you'd like to see what ``SliceBuilders`` are available in Robustness Gym
    today, check out []().
#.  If you're interested in a walkthrough of the ``SliceBuilder`` class in more
    detail, head to [](). Head to []() for a deep dive into the ``CachedOperation``
    class. This is recommended for expert users.
#.  If you'd like to learn more about the motivation behind Robustness Gym, check out
    []().
#.  If you're interested in becoming a contributor, read []().


