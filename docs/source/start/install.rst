
Installation
~~~~~~~~~~~~


**Preamble.** Meerkat's goal is to make it easy for a wide audience to work with *unstructured data* like images, text, audio etc.
We offer a ``pandas``-like DataFrame for unstructured data, as well as primitives for building
graphical user interfaces to query and work with this data across notebooks and web applications. 
Everything is built with first-class support for machine learning models in mind.

Meerkat is great in both notebooks and standalone scripting, and gives you the tools to build full-fledged 
interactive applications that require dealing with messy data and models quickly. 

.. note ::
    Meerkat has particularly strong support for evaluation, error analysis and auditing of ML models, 
    since it was developed initially to support these activities.

**Philosophy.** With Meerkat, we want to give technically minded users the opportunity to extend and 
tinker to customize the library to their liking, while most users can use Meerkat without worrying 
about the details. Ultimately, all our implementation decisions prioritize simplicity, productivity and ergonomics, 
over technical pyrotechnics and bloat.

**Who we are.** Meerkat is being built and maintained by Machine Learning PhD students in the Hazy Research lab at Stanford.
We have varied research backgrounds: we've created new model architectures, studied model robustness and evaluation, 
advanced applications ranging from audio generation to medical imaging, and directly contributed to the 
emergence of Foundation Models. We're excited to build for a future where ML models 
can make it easier to sift and reason through large volumes of data for humans.


.. tab-set::
    .. tab-item:: Main
        
        Meerkat is available on `PyPI <https://pypi.org/project/meerkat-ml/>`_ and can be 
        installed with pip.
    

        .. code-block:: 

            pip install meerkat-ml
            
        
        .. admonition:: Optional dependencies
            
            The core Meerkat package only has a handful of dependencies. Depending on what you 
            use Meerkat for, you may need to install many optional dependencies, since some parts 
            of Meerkat rely on them. 
            
            To install all optional dependencies use: 
            
            .. code-block:: bash
            
                pip install meerkat-ml[all] 
                
            You can also install specific
            groups optional of dependencies using something like: 

            .. code-block:: bash
            
                pip install meerkat-ml[vision,text]
            
            See `setup.py` for a full list of 
            optional dependencies.   

    .. tab-item:: Latest
        
        To install the latest development version of Meerkat use:

        .. code-block:: bash

            pip install "meerkat-ml @ git+https://github.com/robustness-gym/meerkat@dev"

        .. admonition:: Optional Dependencies
        
            Some parts of Meerkat rely on optional dependencies. To install all optional
            dependencies use: 

            .. code-block:: bash

                pip install "meerkat-ml[all] @ git+https://github.com/robustness-gym/meerkat@dev"
            
            You can also install specific groups optional of dependencies using something like: 

            .. code-block:: bash

                ``pip install ``pip install "meerkat-ml[vision,text] @ git+https://github.com/robustness-gym/meerkat@dev"``
                
            See `setup.py` for a full list of optional dependencies.   

    .. tab-item:: Editable

        To install from editable source, clone the meerkat repository and pip install in
        editable mode. 

        .. code-block:: bash

            git clone https://github.com/robustness-gym/meerkat.git
            cd meerkat
            pip install -e .

        .. admonition:: Optional Dependencies
        
            Some parts of Meerkat rely on optional dependencies. To install all optional
            dependencies use: 

            .. code-block:: bash

                pip install -e .[dev]
            
            You can also install specific groups optional of dependencies using something like: 

            .. code-block:: bash

                pip install -e .[vision,text]
                
            See `setup.py` for a full list of optional dependencies.   

