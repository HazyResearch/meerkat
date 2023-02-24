Meerkat users can start interfaces inside Jupyter notebook environments.

.. code-block:: python
    
    import meerkat as mk
    
    # Start the server
    mk.gui.start()
    
    # ...
    # Example: show DataFrame
    df = mk.DataFrame(...)
    df.gui.table() # outputs the table in the notebook
    
