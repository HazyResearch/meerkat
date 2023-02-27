# Reactive Functions

A reactive function in Meerkat is a function that reruns when one of its inputs changes. This allows for creating functions that are responsive to the data they are given.

```{admonition} Definition - _reactive function_
A function that reruns when one of its inputs changes.
```

Reactivity is the essence behind building interactive user interfaces in Meerkat. *Reactive* code is "tracked" by Meerkat so that it can be run again based on if the respective inputs change.

These docs walk through what reactive functions are, how to use them, and how to write your own.

(reactivity_getting_started)=

# Getting Started with Reactive Functions

In any interactive application, certain functions will need to be rerun when states of variables in the application change. Meerkat provides a mechanism for this through the concept of _reactivity_, implemented using the {py:func}`@mk.reactive() <meerkat.reactive()>` decorator.

In these pages, we will discuss

- the concept of reactivity in Meerkat
- how to make code reactive
- common pitfalls and how to avoid them
- a few examples of how to use reactivity
- turning reactivity off

(reactivity/getting-started/reactivity)=

## What is Reactivity?

```{admonition} Definition - _reactive function_
A function that reruns when one of its inputs changes.
```

In Meerkat, reactivity refers to the ability to rerun a function invocation when its inputs change.

The bodies of reactive functions are exactly like regular functions. The difference is that reactive functions are tracked by Meerkat, which handles re-invoking them behind the scenes.

(reactivity/getting-started/reactive-functions)=

## Creating Reactive Functions

Meerkat provides the `@mk.reactive()` decorator to designate a function as reactive. If you haven't used Python decorators before, they are simply wrappers around a function. When we use `@mk.reactive()`, the decorator will return a new function that is reactive.

To use `mk.reactive()`, use standard Python syntax for decorators. There are three ways to do this:

1. Decorate a function with `mk.reactive()`:

   ```python
   @mk.reactive()
   def add_by_1(a):
       return a + 1
   ```

2. Call `mk.reactive()` on a function:

   ```python
   def add_by_1(a):
       return a + 1


   add_by_1 = mk.reactive(add_by_1)
   ```

3. Call `mk.reactive()` on a lambda function:
   ```python
   add_by_1 = mk.reactive(lambda a: a + 1)
   ```

All three methods have the same effect of creating a reactive function called `add_by_1` that can be used in subsequent lines of code.

What does wrapping a function with `mk.reactive()` actually do?
At a high level, these functions will be rerun when their inputs change.

We'll look at this in more detail in a bit. First, though, we'll talk about `Marked` objects, which are intertwined with reactive functions.

(reactivity/getting-started/marked-inputs)=

## Marked Inputs and Reactivity

So far, we learned a simple definition of reactive functions in Meerkat. This definition was almost, but not quite, complete.

Rather than a reactive function being rerun whenever any of its inputs are updated, it is actually rerun whenever any of its **marked** inputs are updated. This is a subtle but important distinction.

**What objects can be marked?** By default, standard objects cannot be marked. However, they can be wrapped in a {class}`~meerkat.Store`. `Store` objects are _markable_, which means we can mark them for use with reactive functions. Python primitives, third-party objects, and custom objects can be wrapped in `Store` objects to make them markable. Other objects in Meerkat like `DataFrame` and `Column` are also markable. All of these objects provide a `.mark()` and `.unmark()` method to control whether they are marked or not.

This means that for a function to react to changes in its inputs, its inputs must be marked. Let's look at a few examples.

```{important}
Inputs into reactive functions must be **marked** for the function to react to changes in them.
```

```python
import meerkat as mk


@mk.reactive()
def add(a, b):
    return a + b


# c1 is updated if either a or b changes
a = mk.Store(1)
b = mk.Store(2)
c1 = add(a, b)

# c2 is updated only if a changes as b is unmarked
a = mk.Store(1)
b = mk.Store(2).unmark()
c2 = add(a, b)

# c3 is updated only if a changes as b is not markable
a = mk.Store(1)
b = 2
c3 = add(a, b)

# c4 will never be updated because neither a nor b are markable
a = 1
b = 2
c4 = add(a, b)
```

Most of the time, we won't need to worry about marking `Store` objects, since they are marked by default.

However, other Meerkat objects like `DataFrame` and `Column` are unmarked by default. If we want reactive functions to react to changes in them, we must call `.mark()` on them prior to passing them to a reactive function.

```python
@mk.reactive()
def df_sum(df):
    return df["a"].sum()


df = mk.DataFrame({"a": [1, 2, 3]})

# c1 is not updated if df changes because DataFrames are unmarked by default
c1 = df_sum(df)

# c2 is updated if df changes as df is now marked
df.mark()
c2 = df_sum(df)
```

Takeaways:

- Any inputs can be passed to reactive functions, but only marked inputs will be used to determine if the function should be rerun.
- Wrap a Python primitive or custom object in `Store` to make it markable.
- Use the `.mark()` and `.unmark()` methods to control whether a markable object is marked or not.
- `Store` objects are marked by default, whereas other Meerkat objects are unmarked by default.

(reactivity/getting-started/chaining-reactive-functions)=

## Chaining Reactive Functions

One of the reasons reactive functions are so powerful is that they can be chained together. This allows us to decompose code into neatly abstracted functions and then compose them together to create complex workflows.

Let's look at an example with two reactive functions, `add` and `multiply`.

```python
@mk.reactive()
def add(a, b):
    return a + b


@mk.reactive()
def multiply(a, b):
    return a * b


x = mk.Store(1)
y = mk.Store(2)

u = add(x, y)  # u is Store(3), type(u) is Store
v = add(y, 7)  # v is Store(9), type(v) is Store
w = multiply(u, 2)  # w is Store(6), type(w) is Store

x.set(4, triggers=True)
# u is now Store(6), type(u) is Store
# v is unchanged
# w is now Store(12), type(w) is Store
```

Let's break down this example:

- `add` and `multiply` are both reactive functions.
- `u` depends on `x` and `y`, so it will be updated if either of them are updated.
- `v` depends only on `y`, so it will only be updated if `y` updates.
- `w` depends on `u`, so it will be updated if `u` is updated.

When we update `x`, only `u` and `w` are updated, by running `add` and `multiply` again respectively. If instead we updated `y`, all three outputs `u`, `v` and `w` would be updated, with `add` running twice and `multiply` running once.

In practice, Meerkat provides in-built reactive functions for many common operations on `Store` objects. So we could have written the example above as:

```python
x = mk.Store(1)
y = mk.Store(2)

u = x + y  # u is Store(3), type(u) is Store
v = y + 7  # v is Store(9), type(v) is Store
w = u * 2  # w is Store(6), type(w) is Store
```

In the above example, the `+` and `*` operators were automatically turned into reactive functions. Other examples include `mk.len()`, `mk.str()`, and `mk.sort()`. You can find a full list of the reactive functions that Meerkat provides {ref}`here <reactivity_inbuilts>`.

(reactivity/getting-started/unmarked)=

## Turning reactivity off

In some cases, we may not want certain code to be reactive. To make code non-reactive, we can use {class}`meerkat.unmarked`.

```{important}
Use `unmarked` to turn off reactivity for a function or block of code.
```

Like `reactive`, `unmarked` can be used as a decorator for functions:

```python
@unmarked()
def add(a, b):
    # This will never return a store because the function is unmarked.
    return a + b


x = mk.Store(1)
y = mk.Store(2)
c = add(x, y)  # c is 3, type(c) is int
```

It can also be used as a context manager:

```python
def add(a, b):
    # This will never return a store because the function is unmarked (by default).
    return a + b


x = mk.Store(1)
y = mk.Store(2)

with unmarked():
    c = add(x, y)  # c is 3, type(c) is int
```

If `reactive` functions are run in an unmarked context, they will not be reactive:

```python
@reactive()
def add(a, b):
    return a + b


x = mk.Store(1)
y = mk.Store(2)

with unmarked():
    c = add(x, y)  # c is 3, type(c) is int
```
