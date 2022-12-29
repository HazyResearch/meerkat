Stores
-------

Most interactive programs require stateful variables, some variables that define the state of a system.

In Meerkat, :class:`Store` allows users to define a stateful variables

- that can be modified by the frontend or backend
- where changes to the variable will trigger operations on the backend
- that can be used as inputs into endpoints or interactive operations


.. figure out how to make these FAQ style dropdowns
What can you put in a :class:`Store`?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Any primitive type or sequence of primitive types (e.g. tuple/list) can be put in a Store.


How do you update a :class:`Store`?
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
A :class:`Store` can be updated by either the backend and/or frontend.

- *Frontend*: If the store is connected to a reactive component on the frontend, changes on the frontend will be reflected on the backend.
- *Backend*: :class:`Store` has a ``.set`` method that can be used to update the value in a reactive way.

.. code-block:: python

    counter = mk.gui.Store(0)
    # Use the special .set() method to update the Store
    counter.set(1)


Faux Paus
^^^^^^^^^
- Setting a store equal to a value (i.e. ``store = 1``) will not update the value of the store. Use ``store.set()``.

.. code-block:: python
    
    # This does not set the value of the store to 1
    counter = mk.gui.Store(0)
    counter = 1
    
    # Use .set to set the value of the store
    counter = mk.gui.Store(0)
    counter.set(1)

- A list of Stores is not reactive, a Store of list is reactive

.. code-block:: python
    
    # my_list is a list of stores. It is not a Store.
    # Operations on my_list will not trigger reactions.
    my_list = [mk.gui.Store(0), mk.gui.Store(0)]
    my_list += [mk.gui.Store(2))]  # this does nothing

    # my_stores is a store containing list of integers.
    # appending will change the value of my_stores.
    # This will trigger reactions.
    my_stores = mk.gui.Store([0, 1])
    my_stores += [2]

- ``Store(None)`` is not ``None``, but is equal to ``None``

.. code-block:: python

    Store(None) is None  # False

    Store(None) == None  # True

- Using shortcut operators (``and``, ``or``, ``not``) with Stores will not return Stores, but using Meerkat's built-in overloads (``mk.cand``, ``mk.cor``, ``mk.cnot``) will

.. code-block:: python

    store = Store("")
    with mk.gui.react():
        # These will not return Stores
        type(store or "default")  # str
        type(store and "default")  # str
        type(not store)  # bool

        # These will return Stores
        type(mk.cor(store, "default"))  # Store
        type(mk.cand(store, "default"))  # Store
        type(mk.cnot(store))  # Store

- Unpacking store of tuples must be done in the ``mk.gui.react()`` to return stores

.. code-block:: python

    @mk.gui.react()
    def add(seq: Tuple[int]):
        return tuple(x + 1 for x in seq)

    store = mk.gui.Store((1, 2))
    # We need to use the `react` decorator here because tuple unpacking
    # happens outside of the function `add`. Without the decorator, the
    # tuple unpacking will not be reactive.
    with mk.gui.react():
        a, b = add(store)
