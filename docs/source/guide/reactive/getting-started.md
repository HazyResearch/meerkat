(reactivity_getting_started)=

# Reactivity

In any interactive application, certain functions will need to be rerun when states of variables in the application change. 
Meerkat provides a mechanism for this through the concept of *reactivity*, implemented using the {py:func}`@mk.reactive() <meerkat.reactive()>` decorator.

In these pages, we will
- explain the concept of reactivity in Meerkat
- walk through how to make your code reactive
- describe common pitfalls and how to avoid them
- show examples of how to use reactivity

(reactivity/getting-started/reactivity)=

## What is Reactivity?

```{admonition} Definition
A _**reactive**_ function re-runs when its inputs change.
```

In Meerkat, reactivity refers to the ability to re-run a function when its inputs change.

Reactive functions are exactly like regular functions, except that they are tracked by Meerkat so that they can be re-run when their inputs change.

(reactivity/getting-started/reactive-functions)=

## Creating Reactive Functions

Meerkat provides the `@mk.reactive()` decorator to designate a function as reactive. If you haven't used a decorator before: it's simply a wrapper around a function. When you use `@mk.reactive()`, the decorator will return a new function that is reactive.

To use `mk.reactive()`, use standard Python syntax for decorators. There are three ways to do this:

##### 1. Decorate a function with `mk.reactive()`
```python
# Use decorator syntax to wrap `add_by_1` in `mk.reactive()`
@mk.reactive()
def add_by_1(a):
    return a + 1
    
# Now use `add_by_1` in your code
add_by_1(...)
```

##### 2. Call `mk.reactive()` on a function
```python
def add_by_1(a):
    return a + 1

add_by_1 = mk.reactive(add_by_1)

# Now use `foo` in your code
add_by_1(...)
```

##### 3. Call `mk.reactive()` on a lambda function
```python
add_by_1 = mk.reactive(lambda a: a + 1)

# Now use `foo` in your code
add_by_1(...)
```

All three methods have the same effect - create a reactive function called `add_by_1`.

What does wrapping a function with `mk.reactive()` actually do? 
At a high level, these functions will be re-run when their inputs change.

We'll look at this in more detail in a bit. First, though, we'll talk about `Marked` objects, which are intertwined with reactive functions.

(reactivity/getting-started/marked-inputs)=

## Marked Inputs and Reactivity
So far, we gave a simple definition of reactive functions in Meerkat. This definition was almost, but not quite, complete.

Rather than a reactive function being re-run whenever any of its inputs are updated, it is actually re-run whenever any of its **marked** inputs are updated. This is a subtle but important distinction.

**What objects can be marked?:** By default, standard objects cannot be marked. However, they can be wrapped in a {py:class}`Store <meerkat.Store>`. `Store` objects are markable, which means we can mark them for use with reactive functions. Python primitives, third-party objects, and custom objects can be wrapped in `Store` objects to make them markable. Other objects in Meerkat like `DataFrame` and `Column` are also markable. All of these objects provide a `.mark()` and `.unmark()` method to control whether they are marked or not.

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

# c4 will never be updated because
# neither a or b are markable objects
a = 1
b = 2
c4 = add(a, b)
```
Most of the time, you won't need to worry about marking `Store` objects, since they are marked by default.

However, other Meerkat objects like `DataFrame` and `Column` are unmarked by default. If you want reactive functions to react to changes in them, you must call `.mark()` on them prior to passing them to a reactive function.

```python
df = mk.DataFrame({"a": [1, 2, 3]})
@mk.reactive()
def df_sum(df):
    return df["a"].sum()

# c1 is not updated if df is updated because 
# df is unmarked by default
c1 = df_sum(df)

# c2 is updated if df is updated as
# df is marked
df.mark()
c2 = df_sum(df)
```


Takeaways:
- Any inputs can be passed to reactive functions, but only marked inputs will be used to determine if the function should be re-run.
- `Store` objects are marked by default, while other Meerkat objects are unmarked by default.
- Wrap Python primitives or custom objects in `Store` to make it markable.

(reactivity/getting-started/chaining-reactive-functions)=

## Chaining Reactive Functions
One of the reasons reactive functions are so powerful is that they can be chained together.
This allows you to break up your code into nicely abstracted functions, and then compose them together to create complex workflows.

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

u = add(x, y) # u is Store(3), type(u) is Store
v = add(y, 7) # v is Store(9), type(v) is Store
w = multiply(u, 2) # w is Store(6), type(w) is Store

x.set(4, triggers=True)
# u is now Store(6), type(u) is Store
# v is not changed
# w is now Store(12), type(w) is Store
```
Let's break down this example:
- `add` and `multiply` are both reactive functions.
- `u` depends on `x` and `y`, so it will be updated if either of them are updated.
- v depends only on `y`, so it will only be updated if `y` updates.
- w depends on `u`, so it will be updated if `u` is updated.

When we update `x`, only `u` and `w` are updated, by running `add` and `multiply` again respectively. If instead we updated `y`, all three outputs `u`, `v` and `w` would be updated, with `add` running twice and `multiply` running once.

In practice, Meerkat provides reactive functions for many common operations on `Store` objects. So we could have written the example above as:
```python
x = mk.Store(1)
y = mk.Store(2)

u = x + y # u is Store(3), type(u) is Store
v = y + 7 # v is Store(9), type(v) is Store
w = u * 2 # w is Store(6), type(w) is Store
```
You can find a full list of the reactive functions that Meerkat provides {ref}`here <reactivity_inbuilts>`.

(reactivity/getting-started/unmarked)=

## Turning reactivity off
In some cases, we may want our code to never be reactive.
This can be quite common when we are writing code that should never be re-run.

To make code, we can use {py:class}`unmarked <meerkat.unmarked>`.

```{important}
Use `unmarked` to turn off reactivity for a function or a block of code.
```


Like `reactive`, `unmarked` can be used as a decorator for functions:
```python
@unmarked()
def add(a, b):
    # This will never return a store, because the function is unmarked.
    return a + b

x = mk.Store(1)
y = mk.Store(2)
c = add(x, y)  # c is 3, type(c) is int
```

It can also be used as a context manager:
```python
def add(a, b):
    # This will never return a store, because the function is unmarked.
    return a + b

x = mk.Store(1)
y = mk.Store(2)

with unmarked():
    c = add(x, y)  # c is 3, type(c) is int
```

If `reactive` functions are run in an unmarked context, they will not be reactive
```python
@reactive()
def add(a, b):
    return a + b

x = mk.Store(1)
y = mk.Store(2)

with unmarked():
    c = add(x, y)  # c is 3, type(c) is int
```