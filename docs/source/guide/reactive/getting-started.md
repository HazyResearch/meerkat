---
file_format: mystnb
kernelspec:
  name: python3
---

# Reactivity

In any interactive application, certain functions will need to be rerun when the state of the application changes. 
Meerkat provides a mechanism for this through the concept of *reactivity*, implemented using the `mk.reactive()` decorator.

This set of pages will
- explain the concept of reactivity in Meerkat
- walk you through how to make your code reactive
- describe common pitfalls and how to avoid them
- show you examples of how to use reactivity


## Reactive Functions

In Meerkat, reactivity refers specifically to the ability to re-run a function when its inputs change. We refer to these functions as *reactive* functions.

Reactive functions are exactly like regular functions, except that they are tracked by Meerkat so that they can be re-run when their inputs change.

### Create a Reactive Function

Meerkat provides the `@mk.reactive()` decorator to designate a function as reactive. If you've never used a decorator before: it's simply a function that takes in a function and returns a new function. When you use `@mk.reactive()`, the decorator will return a new function that is reactive.

To use `mk.reactive()`, use standard Python syntax for decorators. There are three ways to do this:

#### 1. Decorate a function with `mk.reactive()`
```python
# Use decorator syntax to wrap `foo` in `mk.reactive()`
@mk.reactive()
def foo(a):
    return a + 1
    
# Now use `foo` in your code
foo(...)
```

#### 2. Call `mk.reactive()` on a function
```python
def foo(a):
    return a + 1

foo = mk.reactive(foo)

# Now use `foo` in your code
foo(...)
```

#### 3. Call `mk.reactive()` on a lambda function
```python
foo = mk.reactive(lambda a: a + 1)

# Now use `foo` in your code
foo(...)
```

All have the same effect, `foo` is now a reactive function that Meerkat knows about.

What does wrapping a function with `mk.reactive()` actually do? 
At a high level, these functions will be re-run when their inputs change.

We'll look at this in more detail in a bit. First, though, we'll talk about `Store` objects, which are intertwined with reactive functions.

## Recap on Stores
Before we proceed, we'll provide a brief recap on `Store` objects. You can read the full guide on `Store` objects at [XXXXXXXXXXXX]. 

A `Store` is a special object provided by Meerkat that can be used to wrap arbitrary Python objects -- `int`, `str`, `list`, `dict`, `pd.DataFrame`, `pathlib.Path` are just a few examples of objects that can be wrapped by `Store`. 

A major reason to use `Store` objects is that they make it possible for Meerkat to track changes to Python objects.

Let's look at an example to understand why they are so important to reactive functions. 

Let's define a reactive `add` function. This function will take two inputs, `a` and `b`, and return their sum.
```python
@mk.reactive()
def add(a, b):
    return a + b
```

Now, let's try something that might feel natural. Create two `int` objects, call `add` with them, and then change one of them.

```python
x = 1
y = 2
z = add(x, y) # z is 3

x = 4 # z is still 3
```
You might think for a second that `z` should be updated to `6` because `x` changed and `add` is a reactive function. **This is not the case.**

The reason is that `x` is just a variable here. By changing `x`, we aren't changing the object that `x` points to (i.e. the `int` `1`). Instead, we are just changing the variable `x` to point to a different object.

*What we need here is to update the object that `x` points to.* It's impossible to do this with a regular `int`. We can do this with a `Store` instead.

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store

x.set(4, triggers=True) 
# z is now Store(6), type(z) is Store
```
By calling `set()` on `x`, we are changing the object that `x` points to. This is what allows `z` to be updated. (Ignore the `triggers=True` argument.)

`Store` is a transparent wrapper around the object it wraps, so you can use that object as if the `Store` wasn't there.

```python
x = mk.Store(1)
y = mk.Store(2)
z = x + y # z is Store(3), type(z) is Store

message = mk.Store("hello")
message = message + " world" 
# message is Store("hello world"), type(message) is Store
```
A very detailed breakdown of how `Store` objects behave is provided at XXXXXXX. We highly recommend reading that guide.

The takeaways are:
- A `Store` can wrap arbitrary Python objects.
- A `Store` will behave like the object it wraps.
- A `Store` is necessary to track changes when passing an object to a reactive function.


Let's go back to understanding how reactive functions work.

## How do reactive functions work?

Let's take the `add` function again.

**What happens when you wrap `add` with `mk.reactive()`?**
    
```python
@mk.reactive()
def add(a, b):
    return a + b
```

Meerkat tracks the function `add` and re-runs it if any of the inputs to it (i.e. `a` or `b`) change. Internally, Meerkat includes `add` in a computation graph that tracks dependencies between reactive functions.

**What happens when you call `add`?**
```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store
```

Meerkat does two things:
- Unwrap any inputs to `add` that are `Store` objects before passing them into the body of `add`. This allows `add` to be written, and behave like a regular function.
- Take the output of `add`, wrap it in a `Store` object and return it. This lets you use the output of `add` as input to other reactive functions.

**What happens when you change any of the inputs to `add`?**

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store

x.set(4, triggers=True)
# z is now Store(6), type(z) is Store
```

What does Meerkat do when `x` is changed? Let's walk through the steps:
- Meerkat detects that `x` changed because `x` is a `Store`, and `.set()` was called on it.
- Meerkat looks for all reactive functions that were called with `x` as an argument (i.e. `add`).
- Meerkat then re-runs those functions (i.e. `add`) and any functions that depend on their outputs.
- Finally, Meerkat updates the outputs of any reactive functions that it re-runs (i.e. `z`).

Next, let's look at more examples to understand how reactive functions might behave in more complex situations.

## Chaining Reactive Functions
One of the reasons reactive functions are so powerful is that they can be chained together. This allows you to break up your code into nicely abstracted functions, and then compose them together to create complex workflows.

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
You can find a full list of the reactive functions that Meerkat provides at XXXXXXX, and more information about reactivity with `Store` objects at XXXXXXX.

One place where chaining can be helpful is when you want to create a function that takes a DataFrame as input, and returns another DataFrame as output (a "view" of the original DataFrame). Let's look at an example.

```python
@mk.reactive()
def filter(df, column, value):
    return df[df[column] == value]

@mk.reactive()
def sort(df, column):
    return df.sort(column)

@mk.reactive()
def select(df, columns):
    return df[columns]

df = mk.DataFrame({
    "a": [random.randint(1, 10) for _ in range(100)], 
    "b": [random.randint(1, 10) for _ in range(100)],
})
filtered_df = filter(df, "a", 2)
sorted_df = sort(filtered_df, "b")
selected_df = select(sorted_df, ["b"])
```
Here, we filter, sort and select columns from a `mk.DataFrame`, chaining these reactive functions together to achieve this effect. Again, in practice, Meerkat provides reactive functions for many common operations on `mk.DataFrame`, so we could have written the example above as:
```python
filtered_df = df[df["a"] == 2]
sorted_df = filtered_df.sort("b")
selected_df = sorted_df[["b"]]
```

> This pattern works with `mk.DataFrame` and not `pandas.DataFrame`. If you're using a `pd.DataFrame`, we recommend converting to a `mk.DataFrame` using `mk.DataFrame.from_pandas(df)` when using Meerkat's interactive features.
> 
> Alternately, you can also use the `magic` context manager if you want to "turn on" the same functionality for `pd.DataFrame` objects. We recommend reading the guide at XXXXXXX to learn more about this.

## Guidelines for Writing Reactive Functions
Let's go over a few guidelines for writing reactive functions.

An important rule of thumb when using reactive functions is: **don't edit objects inside them without thinking very carefully about the consequences.**

This isn't a hard rule, but it's a good guideline to follow. Let's look at an example that is particularly bad practice:
```python
@mk.reactive()
def foo(df: mk.DataFrame):
    # .... do some stuff to df
    df["a"] += 1
    return df

df = mk.DataFrame({"a": [1, 2, 3]})
df_2 = foo(df)
```
Here, `foo` quietly updates the original `df` object, and then returns it. This is bad practice for a couple of reasons:
- The output of `foo` is the same object as the input to it, which leads to a cyclical dependency for this reactive function. This means that if `df` is updated, `foo` will be re-run, and then `df` will be updated again, and so on.
- Meerkat disregards when `df` is updated in-place inside a reactive function, so it won't update any outputs that depend on `df`. If it did, it would cause an infinite loop (since `foo` must have been called because `df` was updated, and so on).

The way to avoid this is to just not edit objects in-place inside reactive functions. Generally, this will also mean you won't return the same object as the input into the function. Instead, create a new object and return that. For example:
```python
@mk.reactive()
def foo(df: mk.DataFrame):
    # .... do some stuff to df
    df_2 = df.copy()
    df_2["a"] += 1
    return df_2
```
This is a much better way to write `foo`, because it doesn't edit the original `df` object, and it doesn't return the same object as the input into it.

The appropriate place to edit objects in-place in response to user input is inside a `@mk.endpoint()` function, which you can read more about in the guide at XXXXXXX.

You can write reactive functions like any other Python function. Generally, we recommend that you type-hint the inputs without using `Store` objects, since they will be automatically unwrapped inside the function body.

```python
# Do this.
@mk.reactive()
def add(a: int, b: int):
    return a + b

# Don't do this.
@mk.reactive()
def add(a: mk.Store, b: mk.Store):
    return a + b
```






## Using Marked Inputs with Reactive Functions
If we look carefully at the code above, we can see the use a special object called a `Store` to wrap the integer values in `a` and `b`. `Store` objects are **markable** objects, which means we can mark them for use with reactive functions. Other types of objects in Meerkat (like DataFrames, Columns, etc.) are also markable.

> :warning: Reactive functions only react to changes in *marked* inputs.

When a marked object is passed to a reactive function, the function will be re-run if the marked object changes. If an unmarked object is passed to a reactive function, the function will not be re-run if the object changes.

This means that for a function to react to changes in its inputs, the inputs must be marked. Let's look at a few examples

```python
import meerkat as mk

@mk.reactive()
def add(a, b):
    return a + b

# c1 will be updated if a or b changes.
a = mk.Store(1)
b = mk.Store(2)
c = add(a, b)

# c2 will be updated only if a changes
# because b is unmarked
a = mk.Store(1)
b = mk.Store(2).unmark()
c2 = add(a, b)

# c3 will be updatd only if a changes
# because b is not a markable object.
a = mk.Store(1)
b = 2
c3 = add(a, b)

# c4 will not ever be updated because
# neither a or b are markable objects
a = 1
b = 2
c4 = add(a, b)
```

Takeaways:
- Any inputs can be passed to reactive functions, but only marked inputs will be used to determine if the function should be re-run.
- `Store` objects are marked by default.
- Other meerkat objects (e.g. `DataFrame`, `Column`, etc.) are unmarked by default. Make sure to mark them prior to passing them to reactive functions
- All Python objects can be wrapped in a `Store` object to make them markable. This is useful for passing Python objects to reactive functions.

  
# Outline


<!-- 
## How do we make code reactive?

The `mk.reactive()` decorator is used to designate a function is reactive.
These functions will be re-run if the inputs change.
For example, in the code below, if the value of `a` or `b` changes then value of `c` will be automatically updated:

```python
import meerkat as mk

@mk.reactive()
def add(a, b):
    return a + b

a = mk.Store(1)
b = mk.Store(2)
c = add(a, b)
``` -->