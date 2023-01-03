Copy vs. View Behavior 
=======================

In Meerkat, as in other data structures (*e.g.* 
`NumPy <https://numpy.org/doc/stable/user/basics.copies.html>`_, 
`Pandas <https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#indexing-view-versus-copy>`_
), it is important to understand whether or not two variables point to objects that 
share the same underlying data. If they do, modifying one will affect the other. If they
don't, data must be getting copied, which could have implications for efficiency.  
Consider the following example:

.. code-block:: python

   >>> import meerkat as mk
   >>> col1 = mk.TensorColumn(np.arange(10))
   >>> col2 = col1[:4]
   >>> col2[0] = -1
   >>> print(col1[0])

Is ``0`` or ``-1`` printed out? 

It turns out that in this case it is ``-1`` that is 
printed. This is because ``col2`` is a "view" of the ``col1`` array, meaning that 
the two variables point to objects that share the same underlying data. However, if we
were to change the third line to ``col2 = col1[np.arange(4)]``, a seemingly 
inconsequential change, then the underlying data would be copied and it would be ``0`` 
that is printed.

In this guide, we will discuss how to know when two variables in Meerkat share 
underlying data. In general, Meerkat inherits the copy and view behavior of its backend
data structures (Numpy Arrays, Pandas Series, Torch Tensors). So, users who are are '
familiar with those libraries should find it straightforward to predict Meerkat's
copying and viewing behavior. 

We'll begin by defining some terms: coreferences, views and copies. These terms describe
the different relationships that could exist between two variables pointing to 
:class:`~meerkat.Column` or :class:`~meerkat.DataFrame` objects. Then, we'll 
discuss how to know whether indexing a Meerkat data structures will result in a copy, 
coreference or view.

Copies, Views, and Coreferences
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Columns
-------

Let’s enumerate the different relationships that could
exist between two column variables ``col1`` and ``col2``.

**Coreferences -** Both variables refer to the same :class:`~meerkat.Column`
object.

.. code:: python

   >>> col1 is col2
   True

Of course, in this case, anything changes made to ``col1`` will also be
made to ``col2`` and vice versa.

**Views -** The variables refer to different :class:`~meerkat.Column` objects
(*i.e.* ``col1 is not col1``), but modifying the data of ``col1``
affects ``col2`` :

1. either because ``col1.data`` and ``col2.data`` reference the same
   object

   .. code:: python

      # a. the underlying data variables reference the same object 
      >>> col1.data is col2.data
      True

2. or because ``col1.data`` is a view of ``col2.data`` (or vice versa)

   .. code:: python

      ## For example, if col1.data is np.ndarray
      >>> isinstance(col1.data, np.ndarray)
      True
      # b. the underlying data share memory
      >>> col1.data.base is col2.data.base
      True 

-  *How are views created?* Views of a column are created in one of two
   ways:

   1. Implicitly with ``col._clone(data=new_data)`` where ``col.data``
      shares memory with ``new_data``\ for one of the reasons described
      above.
   2. Explicitly with ``col.view()`` which is simply a wrapper around
      ``col._clone``:

      .. code:: python

         def view(self):
             return self._clone()

-  *What about other attributes?* (*e.g.* ``loader`` in an
   ``ImageColumn``) It depends.

   ``col1`` and ``col2`` refer to different column objects, so
   assignment to attributes in ``col1`` will not affect ``col2`` (and
   vice versa):

   .. code:: python

      >>> col1.loader = fn1
      >>> col1.loader == col2.loader
      False

   However, these attributes are not copied! So, stateful changes to the
   attributes will carry across columns:

   .. code:: python

      >>> col1.loader.size = 224
      >>> col2.loader.size == 224
      True

   If we’d like attributes, we’ll have to use "*Deep Copies".*

**Copies**\ *–* The variables refer to different :class:`~meerkat.Column`
objects (*i.e.* ``col1 is not col1``), and modifying the data of
``col1`` does **not** affect ``col2``

In this case, ``col1.data`` and ``[col2.data](http://col2.data)`` do not
share memory.

-  *How are copies created?* Copies of a column are created in one of
   two ways:

   1. Implicitly with ``col._clone(data=new_data)`` where
      ``[col.data](http://col.data)`` does not share memory with
      ``new_data``.
   2. Explicitly with ``col.copy()`` which is simply a wrapper around
      ``col._clone``:

      .. code:: python

         def copy(self):
             new_data = self._copy_data()
             return self._clone(data=new_data)

      where ``_copy_data`` is a backend-specific method that copies the
      data. For example, if the backend is a Numpy Array, then
      ``_copy_data`` will simply ``return self.data.copy()``. This is an
      important point: each column must know how to truly copy it’s
      data.

-  *What about other attributes?* (*e.g.* ``loader`` in an
   ``ImageColumn``) Same as “View” above.

DataFrames
----------

Let’s do the same for two DataFrame variables ``df1`` and ``df2``.

**Coreferences -** Both variables refer to the same ``DataFrame``
object.

.. code:: python

   >>> df1 is df2
   True

Of course, in this case, anything that is done to ``df1`` will also be
done to ``df2`` and vice versa.

**Views -** The variables refer to different ``DataFrame`` objects
(*i.e.* ``df1 is not df2``), but some of the columns in ``df1`` are
`coreferences <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
or
`views <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
of some of the columns in ``df2``

-  *How are views created? Views* of a DataFrame are created in one of
   three ways:

   1. Implicitly with ``df._clone(data=new_data)`` where ``df.columns``
      includes some columns with ``new_data``\ for one of the reasons
      described above.
   2. Implicitly when a column from one DataFrame is added to another
      (*e.g.* ``df1["a"] = df2["b"]``. Behind the scenes,
   3. Explicitly with ``df.view()`` which simply calls ``col.view()`` on
      all its columns and then passes them
      ``df._clone(data=view_columns)``

-  *What about other attributes?* (*e.g.* ``index_column`` in an
   ``EntityDataFrame``) It depends.

   ``df1`` and ``df2`` refer to different column objects, so assignment
   to attributes in ``df1`` will not affect ``df2`` (and vice versa):

   .. code:: python

      >>> df1.loader = fn1
      >>> df1.loader == df2.loader
      False

   However, these attributes are not copied! So, stateful changes to the
   attributes will carry across DataFrames:

   .. code:: python

      >>> df1.loader.size = 224
      >>> df2.loader.size == 224
      True

**Copies**\ *–* The variables refer to different ``DataFrame`` objects
(*i.e.* ``df1 is not df2``), and all of the columns in ``df1`` are
copies of the the columns in ``df2``

-  *How are copies created?* Copies of a column are created in one of
   two ways.

   1. Implicitly with ``col._clone(data=new_data)`` where
      ``[col.data](http://col.data)`` does not share memory with
      ``new_data``.
   2. Explicitly with ``col.copy()`` which is simply a wrapper around
      ``col._clone``:

      .. code:: python

         def copy(self):
             new_data = self._copy_data()
             return self._clone(data=new_data)

      where ``_copy_data`` is a backend-specific method that copies the
      data. For example, if the backend is a Numpy Array, then
      ``_copy_data`` will simply ``return self.data.copy()``. This is an
      important point: each column must know how to truly copy it’s
      data.

-  *What about other attributes?* (*e.g.* ``index_column`` in an
   ``EntityDataFrame``) Same as “View” above.

Behavior when Indexing
~~~~~~~~~~~~~~~~~~~~~~~

Indexing rows
--------------

In Meerkat, we select rows by indexing with ``int``, ``slice`` ,
``Sequence[int]``, or an ``np.ndarray`` , ``torch.Tensor``,
``pandas.Series`` with an integer or boolean type.

We can select rows from an :class:`~meerkat.Column`\ …

.. code:: python

   col: mk.Column = ...
   # (1) int -> single value
   value: object = col[0] 
   # (2) slice -> a sub column
   new_col: mk.Column = col[0:10]
   # (3) sequence -> a sub column
   new_col: mk.Column = col[[0, 4, 6]]

… or from a ``DataFrame``

.. code:: python

   df: mk.DataFrame = ...
   # (1) int -> dict
   row: dict = df[0] 
   # (2) slice -> a DataFrame slice
   new_df: mk.DataFrame = df[0:10]
   # (3) sequence -> a DataFrame slice
   new_df: mk.Datapanel = df[[0, 4, 6]]

**From a column.** When selecting rows from a column ``col``, Meerkat
takes the following approach:

**Step 1.** Indexes the underlying data object stored at
``[col.data](http://col.data)`` (*e.g.* ``np.ndarray`` or
``torch.tensor``) *always* deferring to the copy/view strategy of that
data structure. This gives us a new data object, ``new_data`` which may
or may not share memory with with the original ``col.data`` depending on
the strategy of the underlying data structure.

-  Copy/View strategies of data structures underlying core Meerkat
   columns.

   -  **torch**

         When accessing the contents of a tensor via indexing, PyTorch
         follows Numpy behaviors that basic indexing returns views,
         while advanced indexing returns a copy. Assignment via either
         basic or advanced indexing is in-place. See more examples in
         `Numpy indexing
         documentation <https://numpy.org/doc/stable/reference/arrays.indexing.html>`__.

   -  **numpy**

         Advanced indexing always returns a copy of the data (contrast
         with basic slicing that returns a view).
         (`source <https://numpy.org/doc/stable/reference/arrays.indexing.html>`__)

   -  **pandas**

         But in pandas, whether you get a view or not depends on the
         structure of the DataFrame and, if you are trying to modify a
         slice, the nature of the modification.
         (`source <https://www.practicaldatascience.org/html/views_and_copies_in_pandas.html>`__)


**Step 2.**
`Clones <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
the original column, ``col``, and stores the the newly indexed data
object, ``new_data``, in it (*i.e.* with ``col._clone(data=new_data)``.

So, selecting rows from a column ``col`` returns either a
`view <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
or a
`copy <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__,
depending on the underlying data structure.

**From a DataFrame.** When selecting rows from a DataFrame ``df``,
Meerkat takes the following approach:

**Step 1.** Indexes each of the columns using the strategy above.

Note: sometimes this step proceeds in batches according to the
BlockManager.

**Step 2.**
`Clones <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
the original DataFrame, ``df``, passing the newly indexed columns. This
new DataFrame will be:

-  either a
   `view <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
   of the original ``df``, if any of the indexed columns are views
-  or a copy if all of the indexed columns are copies

Indexing columns
-----------------

In Meerkat, we select columns from a ``DataFrame`` by either indexing
with ``str`` or a ``Sequence[str]`` :

.. code:: python

   # (1) `str` -> single column
   col: mk.Column = df["col_a"]
   # (2) `Sequence[str]` -> multiple columns
   df: mk.DataFrame = df[["col_a", "col_b"]]

When selecting columns from a ``DataFrame``, Meerkat **always** returns
a
`coreference <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
to the underlying column(s) – *not* a copy or view.

(1) Indexing a single column (*i.e.* with a ``str``) returns the
    underlying :class:`~meerkat.Column` object directly. In the example below
    ``col1`` and ``col2`` are
    `coreferences <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
    of the same column.

.. code:: python

   # (1) `str` -> single column
   >>> col1: mk.Column = df["col_a"]
   >>> col2: mk.Column = df["col_a"]
   >>> col1 is col2
   True

(2) Indexing multiple columns (*i.e.* with ``Sequence[str]``) returns a
    `view <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
    of the ``DataFrame`` holding
    `coreferences <https://www.notion.so/meerkat-working-doc-40d70d094ac0495684d3fd8ddc809343>`__
    to the columns in the original ``DataFrame``. This means the
    :class:`~meerkat.Column` objects held in the new ``DataFrame`` are the
    same :class:`~meerkat.Column` objects held in the original ``DataFrame``.

.. code:: python

   # (1) `Sequence[str]` -> single column
   >>> new_df: mk.DataFrame = df[["col_a", "col_b"]]
   >>> new_df["col_a"] is df["col_a"]
   True
   >>> new_df["col_a"].data is df["col_a"].data
   True
