#####################
Endpoints vs. Reactive Functions
#####################

Why do we need both endpoint functions (``endpoint`` decorator) and reactive functions (``reactive`` decorator)?

Endpoints are functions that can be run by the frontend in order to update the state of the application e.g. update ``Store`` and ``DataFrame`` objects.
Because of this, it's actually possible to write every Meerkat application using only endpoints, since we can arbitrarily manipulate state with them. 
This is actually similar to the design of libraries like Gradio, which do not have a concept of reactive functions.

Then why have reactive functions at all? The main reason is programmer efficiency and ergonomics, as we explain on this page. 

Dependent State Variables
-------------------------

Suppose we have an application state that has dependent variables in it. This is common in data apps, where we might have many ``DataFrame`` objects that are views of another ``DataFrame``.

All the views in this case are dependent on the original ``DataFrame`` through some transformation. If we were only allowed to use endpoints, then we would have to write a separate endpoint for each view, and then call each endpoint whenever we needed to update the state. This is not ideal, since it's a lot of boilerplate code, and there's a high chance of making mistakes by manually maintaining the relationship between these ``DataFrame`` objects.

Here's a simple example that illustrates this with ``Store`` objects.

.. code-block:: python
    
    import meerkat as mk
    
    def add_one(x):
        return x + 1
        
    def add_two(x):
        return x + 2
        
    # State variables
    a = mk.Store(1)
    ap = mk.Store(1)
    
    # Dependent state variables
    b = add_one(a)
    c = add_two(a)
    d = add_one(b)
    
    bp = add_one(ap)
    cp = add_two(ap)
    dp = add_one(bp)
    
    # Now we need to define two endpoints, one for a and one for ap
    @mk.endpoint()
    def foo_a():
        a.set(a + 1)
        # We have to manually ensure all dependent state variables are updated
        # ..in every endpoint that updates a state variable
        b.set(add_one(a))
        c.set(add_two(a))
        d.set(add_one(b))
        
    @mk.endpoint()
    def foo_ap():
        ap.set(ap + 1)
        # And again...
        bp.set(add_one(ap))
        cp.set(add_two(ap))
        dp.set(add_one(bp))
        
It's clear there are a few problems here:

1. We're writing ``.set()`` statements for every dependent state variable, which is a lot of boilerplate code.
2. We're manually keeping track of the relationship between the state variables, which is error-prone e.g. we need to remember to update ``d`` which indirectly depends on ``a`` through ``b``.
3. We can't write a single endpoint that updates both ``a`` and ``ap`` because we don't know which dependent state variables to update.
    
This is a nightmare to program, and we can do better! 

Reactive Functions to the Rescue
--------------------------------

Here's the same example with reactive functions.

.. code-block:: python
    
    import meerkat as mk
    
    # Designate the functions as reactive
    @mk.reactive()
    def add_one(x):
        return x + 1
        
    @mk.reactive()
    def add_two(x):
        return x + 2
        
    # State variables
    a = mk.Store(1)
    ap = mk.Store(1)
    
    # Dependent state variables
    b = add_one(a)
    c = add_two(a)
    d = add_one(b)
    
    bp = add_one(ap)
    cp = add_two(ap)
    dp = add_one(bp)
    
    @mk.endpoint()
    def foo(x: mk.Store):
        x.set(x + 1)
        # Now we don't have to manually ensure all dependent state variables are updated
        # because the reactive functions will automatically re-run when necessary

This solution is more pleasant to write: no extra ``.set()`` statements, dependent state variables are automatically updated, and we can write a single endpoint that updates both ``a`` and ``ap``.


Selective Updates
-----------------
Ractive functions only update the dependent variables that *need* to be updated. Internally, Meerkat keeps track of the dependency graph between state variables and reactive functions, and only re-runs the reactive functions that depend on the state variables that have changed. If you update ``a``, then only the reactive functions that depend on ``a`` will be re-run.


Statements
----------
Another great feature of reactivity in Meerkat is the ability to write reactive statements. This is best illustrated with a slight rework of the previous example.

.. code-block:: python
    
    import meerkat as mk
    
    # State variables
    a = mk.Store(1)
    ap = mk.Store(1)
    
    # Dependent state variables
    # These statements will behave like reactive functions!
    b = a + 1
    c = a + 2
    d = b + 1
    
    bp = ap + 1
    cp = ap + 2
    dp = bp + 1
        
    
    @mk.endpoint()
    def foo(x: mk.Store):
        x.set(x + 1)

This is a very powerful feature, since it allows us to write reactive code in a natural way. It's common to have code in Jupyter notebooks that constructs views of a ``DataFrame``. With reactive statements, we can easily convert this code into a Meerkat app.

Final Thoughts
--------------

Reactive functions are a powerful feature of Meerkat that improve programmer efficiency, and provide a natural way to wrap existing data science code into interactive applications. We hope you'll find them useful in your apps.
