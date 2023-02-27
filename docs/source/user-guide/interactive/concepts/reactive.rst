Reactivity in Meerkat
=====================

In any interface, certain operations need to be performed again if the state of the program changes.
More specifically, if the inputs into that operation change, we want to re-run that operation.

Reactivity is the essence behind building interactive user interfaces in Meerkat.
*Reactive* code is "tracked" so that it can be run again based on if the respective inputs change.

How do we make code reactive?
-----------------------------
`mk.gui.react` is used to manage reactivity in Meerkat.
Code that is wrapped in `mk.gui.react` is tracked and re-run if the inputs change.
For example, in the code below, if the value of `a` or `b` changes then value of `c` will be automatically updated:

.. code-block:: python

    a = mk.Store(1)
    b = mk.Store(2)
    with mk.gui.react():
        c = a + b

`mk.gui.react` can be used as a decorator to make a function reactive.
If either inputs to the function (`a` or `b`) change, the `add` function will be re-run and the value of `c` will be automatically updated.

.. code-block:: python

    @mk.gui.react()
    def add(a, b):
        return a + b
    
    a = mk.Store(a)
    b = mk.Store(b)
    c = add(a, b)  # this operation will be tracked

Is all code inside `mk.gui.react` reactive?
-------------------------------------------
No, only code that uses reactive types (`DataFrames` and `Stores`) will be tracked.
For example, the code below will not be tracked:

.. code-block:: python

    a = 1
    b = 2
    with mk.gui.react():
        c = a + b  # this is the sum of two integers, it will not be tracked

If a function is decorated, you should pass in reactive types as inputs.
Inputs of primitive types (int, str, etc.) will be auto-wrapped as stores, but this is not recommended.

Can I turn reactivity off?
--------------------------
Yes, you can force code to not be reactive by using `mk.gui.no_react(False)`:

.. code-block:: python

    a = mk.Store(1)
    b = mk.Store(2)
    with mk.gui.react():
        c = a + b  # this is reactive
        with mk.gui.react(False):
            d = a * b  # this is not reactive

(Advanced) Passive reactivity
-----------------------------
Sometimes, we may want a function to be *passively reactive*:
if reactivity is on, the function should be reactive. If it is off, then it's not reactive.

To make a function passively reactive, use the `mk.gui.reactive` decorator:

.. code-block:: python

    @mk.gui.reactive
    def add(a, b):
        return a + b
    
    a = mk.Store(1)
    b = mk.Store(2)
    c = add(a, b)  # this operation will not be tracked because reactivity is off by default
    with mk.gui.react(True):
        d = add(a, b)  # this operation will be tracked because we explicitly turned reactivity on

