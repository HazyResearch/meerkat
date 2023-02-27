(reactivity_faqs)=

# FAQs

## How do we make code reactive?

<!-- #FIXME this is out of date -->

`mk.reactive` is used to manage reactivity in Meerkat.
Code that is wrapped in `mk.reactive` is tracked and re-run if the inputs change.
For example, in the code below, if the value of `a` or `b` changes then value of `c` will be automatically updated:

```python
a = mk.Store(1)
b = mk.Store(2)
with mk.reactive():
    c = a + b
```

`mk.reactive` can be used as a decorator to make a function reactive.
If either inputs to the function (`a` or `b`) change, the `add` function will be re-run and the value of `c` will be automatically updated.

```python
@mk.reactive()
def add(a, b):
    return a + b

a = mk.Store(a)
b = mk.Store(b)
c = add(a, b)  # this operation will be tracked
```

## Is all code inside `with mk.reactive()` reactive?

No, only code that uses reactive types (`DataFrames` and `Stores`) will be tracked.
For example, the code below will not be tracked:

```python
a = 1
b = 2
with mk.reactive():
    c = a + b  # this is the sum of two integers, it will not be tracked
```

If a function is decorated, you should pass in reactive types as inputs.
Inputs of primitive types (int, str, etc.) will be auto-wrapped as stores, but this is not recommended.

## How do we turn off reactivity?

We can wrap code with the `@unmarked()` decorator or the `with unmarked()` context manager.

## How do we create a _passively_ reactive function?

<!-- #FIXME out of date -->

Sometimes, we may want a function to be _passively reactive_:
if reactivity is on, the function should be reactive. If it is off, then it's not reactive.

To make a function passively reactive, use the `mk.reactiveive` decorator:

```python
@mk.reactive()
def add(a, b):
    return a + b


a = mk.Store(1)
b = mk.Store(2)

c = add(a, b)  # this operation will not be tracked because reactivity is off by default

with mk.reactive(True):
    d = add(a, b)  # this operation will be tracked because we explicitly turned reactivity on
```
