Configuring Meerkat
====================

Several aspects of Meerkat's behavior can be configured by the user. 
For example, one may wish to change the number of DataPanel rows shown in Jupyter 
Notebooks.

You can see the current state of the Meerkat configuration with:

.. ipython:: python

    import meerkat as mk
    mk.config    

Configuring with YAML
----------------------

To make persistent changes to the configuration, edit the YAML file at `~/.meerkat/config.yaml`. 
For example, the YAML file below will change the default directory to which datasets are downloaded and increase the max number of rows displayed in Jupyter Notebooks: 

.. code-block:: yaml

    dataset:
        root_dir: "/path/to/storage"
    display:
        max_rows: 20

If you would rather keep the YAML file elsewhere, then you can set the environment variable 
``MEERKAT_CONFIG`` to point to the file:

.. code-block:: bash

    export MEERKAT_CONFIG="/path/to/mk/config.yaml"

If you're using a conda, you can permanently set this variable for your environment:

.. code-block:: bash

    conda env config vars set MEERKAT_CONFIG="path/to/mk/config.yaml"
    conda activate env_name  # need to reactivate the environment 


Configuring Programmatically
------------------------------

You can also update the config programmatically, though, unlike the YAML method above, these changes will not persist beyond the lifetime of your program. 

.. code-block:: python

    mk.config.datasets.root_dir = "/path/to/storage"
    mk.config.public_bucket_name = "mk-test"

