Running Meerkat Interfaces
==========================

Meerkat interfaces can be run as standalone applications or embedded in Jupyter notebooks. 
This page describes the different ways to run Meerkat interfaces.


Running Interfaces in Scripts
-----------------------------

The easiest way to run Meerkat interfaces is to use the ``mk`` command line tool. 
This comes pre-installed with the Meerkat Python package, and can be used to run
scripts that create Meerkat interfaces.

To run a script, use the ``mk run`` command:

.. code-block:: bash

    mk run my_script.py
    # modules are fine too: mk run module.that.contains.my_script
    
.. tip::
    
    By default, the ``mk run`` command runs in development (``dev``) mode. This means that the
    script is re-run every time you save a change to the script (live-reloading).
    
    To run in production (``prod``) mode, use the ``--prod`` flag:
    
    .. code-block:: bash
    
        mk run my_script.py --prod
        
    Read more about the ``mk`` command line tool in the :doc:`/reference/cli` reference.


**Requirements.** The only requirement is that the script create an ``Interface`` object that
is assigned to a variable i.e. your script should look something like this:

.. code-block:: python
    
    import meerkat as mk
    
    # ....
    
    interface = mk.gui.Interface(...)


Meerkat assumes that this variable is called ``interface`` by default. 
If you use a different variable name, pass it as an
argument to the ``mk run`` command:

.. code-block:: bash

    mk run my_script.py --target <variable_name>


.. note::
    
    The ``mk run`` command does a few things behind the scenes:
    
    1. It creates a FastAPI server that runs the script, and serves the backend API.
    2. It creates a frontend server to serve the web interface.
    3. It pipes the outputs of your script, and the logs of the servers to the terminal.


Interfaces in Jupyter Notebooks
-------------------------------

Meerkat interfaces can also be run in Jupyter notebooks. This is useful if you want to
view one or more interfaces in output cells, e.g. to view a multimodal dataset or 
produce an interesting visualization.

To run an interface in a Jupyter notebook, Meerkat provides the ``start`` function in the
``meerkat.gui`` module. This function takes care of setting up the necessary servers.

.. code-block:: python
    
    # Do this once at the top of your notebook
    import meerkat as mk
    mk.gui.start()


Then, whenever you want to run an interface, simply create an ``Interface`` object in 
a code cell, and it will be displayed in the output cell.

.. code-block:: python
    
    interface = mk.gui.Interface(...)


.. tip::
    
    Meerkat DataFrames come built in with a ``gui`` namespace that contains a collection of
    useful interfaces for notebooks. For example, to view a DataFrame in a notebook as a 
    table, simply call the ``table`` method:
    
    .. code-block:: python
    
        df.gui.table()
        
    To view it as a gallery, call the ``gallery`` method:
    
    .. code-block:: python
    
        df.gui.gallery()
        
    To see the full list of interfaces, see the :doc:`/reference/dataframe` reference.


.. warning::
    
    Interactions with interfaces in output cells affect the state of the notebook. This is good to 
    keep in mind when running interfaces for labeling or otherwise modifying data.
    

The ``start`` Function for Scripting
------------------------------------

The ``start`` function can also be used to run Meerkat interfaces in scripts. While we recommend
using the ``mk run`` command in most cases, there are a few situations where you might want to use
``start`` instead:

1. You want to set breakpoints in your script for debugging. ``mk run`` does not support this.
2. You want to have access to a Python interpreter while the script is running. By default, ``start``
   will block the current thread, and keep a Python interpreter running so you can interact with 
   the script. 

To use this method, call the ``start`` function in your script before creating an interface.

.. code-block:: python

    import meerkat as mk
    
    # ....
    
    mk.gui.start()
    interface = mk.gui.Interface(...)
    
To run the script, use the ``python`` command directly (not ``mk run``)

.. code-block:: bash

    python my_script.py
    
.. attention::
    
    Using ``start`` does not support live-reloading. If you want to use live-reloading, use the
    ``mk run`` command instead.
