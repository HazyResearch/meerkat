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
- Assigning a value to a store variable with Python assingment (i.e. `=`)

.. code-block:: python
    
    counter = mk.gui.Store(0)
    # This does not set the value of the store to 1
    counter = 1

- A list of Stores is not reactive, a Store of list is reactive

.. code-block:: python
    
    # my_list is a list of stores. It is not a Store.
    # Operations on my_list will not trigger reactions.
    my_list = [mk.gui.Store(0), mk.gui.Store(0)]
    my_list.append(mk.gui.Store(2))  # this does nothing

    # my_stores is a store containing list of integers.
    # appending will change the value of my_stores.
    # This will trigger reactions.
    my_stores = mk.gui.Store([mk.gui.Store(0), mk.gui.Store(1)])
    my_stores.append(mk.gui.Store(2))