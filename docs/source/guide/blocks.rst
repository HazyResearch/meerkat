
DataFrame Internals: Blocks
============================
In Meerkat, the columns of a DataFrame are grouped together into *blocks*, sets of columns with similar underlying storage (*e.g.* NumPy arrays). Organizing columns into blocks enables:

1. Vectorized row-wise operations (*e.g.* slicing, reduction)
2. Simplified I/O and improved latency

The most important internal piece of the Meerkat :class:`~meerkat.DataFrame` implementation is the :class:`~meerkat.block.manager.BlockManager`, a dict-like object that maps column names to columns. The  :class:`~meerkat.block.manager.BlockManager` manages links between a DataFrame's columns and data blocks (`AbstractBlock`, `NumpyBlock`) where the data is actually stored. It implements `consolidate`, which takes columns of similar type in a DataFrame and stores their data together in a block, and `apply` which applies row-wise operations (e.g. __getitem__) to the blocks in a vectorized fashion. Other important classes:  

- :class:`~meerkat.block.ref.BlockRef` objects link a block with the  :class:`~meerkat.block.manager.BlockManager`. These are critical to the functioning of the BlockManager and are the primary type of object passed between the blocks and the block manager. They consists of two things:

  1. A reference to the block (`self.block`)
  2. A set of columns in the :class:`~meerkat.block.manager.BlockManager` whose data live in the `Block`
- :class:`~meerkat.mixins.blockable.BlockableMixin` - a mixin used with `AbstractColumn` that holds references to a column's block and the columns index in the block
- :class:`~meerkat.block.abstract.BlockView` - a simple DataClass holding a block and an index into the block. It is typical for new columns to be created from `BlockView`


:class:`~meerkat.block.manager.BlockManager`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Manages all the columns in a :class:`~meerkat.DataFrame` and holds references
(:class:`~meerkat.block.ref.BlockRef`) to all the blocks in a :class:`~meerkat.DataFrame`. This is done with
two collections:

-  ``_columns``, a dictionary mapping from column names to
   ``AbstractColumn``
-  ``_block_refs``, a dictionary mapping from the blocks id to
   ``BlockRef``

Implement the following methods:

**``consolidate``**

.. code:: python

   ### PSEUDOCODE
   block_groups = group blocks by signature
   for group in block_groups:
       for block in group:
           # get a "view" of the subset of the columns in the block 
       # (note this may take multiple )
       # concat the blocks and get mapping from name
     # and figure out the mapping of columns to index in block 
       

IMPORTANT: After a consolidate, all columns have their own memory!

``**apply**``

How do block operations work?

-  Apply the operation to each block in the data panel,

   -  Each new block should

-  Create mapping

``**add**``

-  Single
-  Multiple

``**remove**``

When deleting a column we have to be sure to delete the reference to the
block \***\*

``get_columns``

``BlockRef``
~~~~~~~~~~~~

A ``BlockRef`` is the link between a DataFrame and a single block. It
consists of two things:

-  A reference to the block (``self._block``)
-  A set of columns (of type\ ``BlockableMixin``

``AbstractBlock``
~~~~~~~~~~~~~~~~~
Multiple A block can exist in multiple . 

``BlockableMixin``
~~~~~~~~~~~~~~~~~~

This is mixed into ``AbstractColumn`` subclasses that can take part of a
block (*e.g.*
