(reactivity_concepts_stores)=

## Recap: Stores

Recall, a core principle of reactivity is that Meerkat only tracks **marked** inputs into functions. If an input is not marked, Meerkat will not track it and will not rerun the function if it changes.

The defacto way to mark an object is to wrap it in a {class}`~meerkat.Store`. In this section, we will briefly cover the importance of the `Store` object. You can read the full `Store` guide {ref}`here <guide_store_getting_started>`.

A `Store` is a special object provided by Meerkat that can be used to wrap arbitrary Python objects such as primitive types (e.g., `int`, `str`, `list`, `dict`, etc.), third-party objects (e.g., {py:class}`pandas.DataFrame`, `pathlib.Path`), and even our own custom objects. `Store` objects make it possible for Meerkat to track changes to Python objects.

Let's look at an example to understand why they are so important to reactive functions.

```python
@mk.reactive()
def add(a, b):
    return a + b
```

Now, let's try something that might feel natural. We'll create two `int` objects, call `add` with them, and then change one of them.

```python
x = 1
y = 2
z = add(x, y)  # z is 3

x = 4  # z is still 3
```

You might think for a second that `z` should be updated to `6` because `x` changed and `add` is a reactive function. **This is not the case.**

This is because `x` is just an `int`. By changing `x`, we aren't changing the object that `x` points to (i.e., the `int` `1`). Instead, we are just changing the variable `x` to point to a different object.

_We need to update the object that `x` points to._ It's impossible to do this with a regular `int`. We can do this with a `Store` instead.

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y)  # z is Store(3), type(z) is Store

x.set(4, triggers=True)
# z is now Store(6), type(z) is Store
```

By calling `set()` on `x`, we are changing the object that `x` points to. This is what allows `z` to be updated.

```{note}
`triggers=True` is an optional argument to `Store.set()` to trigger the reactive chain. It should never be called explicitly in the code but can be helpful for debugging purposes.
<!-- #FIXME - why are we calling it explicitly then? -->
```

`Store` is a _transparent wrapper_ around the object it wraps, so we can use that object as if the `Store` wasn't there.

```python
x = mk.Store(1)
y = mk.Store(2)
z = x + y  # z is Store(3), type(z) is Store

message = mk.Store("hello")
message = message + " world"  # message is Store("hello world"), type(message) is Store
```

We highly recommend reading the detailed breakdown of how `Store` objects behave, which can be found here XXXXXXX (#FIXME).

The takeaways are:

- A `Store` can wrap arbitrary Python objects.
- A `Store` will behave like the object it wraps.
- A `Store` is necessary to track changes when passing an object to a reactive function.
