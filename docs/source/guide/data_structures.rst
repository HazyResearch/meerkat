
Introduction to Data Structures 
================================

Meerkat provides two data structures, the column and the datapanel, that together help 
you  wrangle their data. Everything you do with Meerkat will 
involve one or both of these data structures, so we begin this user guide with their
high-level introduction. 


Column
-------
A column is a sequential data structure (analagous to a `Series <https://pandas.pydata.org/docs/reference/api/pandas.Series.html>`_ in Pandas or a `Vector <https://cran.r-project.org/doc/manuals/r-release/R-intro.html#Simple-manipulations-numbers-and-vectors>`_ in R). 
Meerkat supports a diverse set of column types (*e.g.* :class:`~meerkat.NumpyArrayColumn`, 
:class:`~meerkat.ImageColumn`), each with its own backend for storage. 
All columns are subclasses of :class:`~meerkat.AbstractColumn` and share a common 
interface, which includes :meth:`~meerkat.AbstractColumn.__len__`, :meth:`~meerkat.AbstractColumn.__getitem__`, :meth:`~meerkat.AbstractColumn.__setitem__`, :meth:`~meerkat.AbstractColumn.filter`, :meth:`~meerkat.AbstractColumn.map`, and :meth:`~meerkat.AbstractColumn.concat`.

Some column types may also have additional functionality. For example, 
:class:`~meerkat.NumpyArrayColumn` inherits most of the functionality of an
`ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`_.


DataPanel
----------
A :class:`DataPanel` is a collection of equal-length columns (analagous to a `DataFrame <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html#pandas.DataFrame>`_ in Pandas or R). 