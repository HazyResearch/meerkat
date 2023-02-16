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

#### 1. Decorate a Function with `mk.reactive()`
```python
# Use decorator syntax to wrap `foo` in `mk.reactive()`
@mk.reactive()
def foo(a):
    return a + 1
    
# Now use `foo` in your code
foo(...)
```

#### 2. Call `mk.reactive()` on a Function
```python
def foo(a):
    return a + 1

foo = mk.reactive(foo)

# Now use `foo` in your code
foo(...)
```

#### 3. Call `mk.reactive()` on a Lambda Function
```python
foo = mk.reactive(lambda a: a + 1)

# Now use `foo` in your code
foo(...)
```

All have the same effect, `foo` is now a reactive function that Meerkat knows about.

What does wrapping a function with `mk.reactive()` actually do? We'll talk through this in more detail next. 

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

*What we need here is to update the object that `x` points to.* It's impossible to do this with a regular `int` object. But we can do this with a `Store` object.

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store

x.set(4, triggers=True) 
# z is now Store(6), type(z) is Store
```
By calling `set()` on `x`, we are changing the object that `x` points to. This is what allows `z` to be updated. (Ignore the `triggers=True` argument for now. We'll talk about it later.)

`Store` is a transparent wrapper around the object it wraps, so you can use that object as if the `Store` wasn't there.

```python
x = mk.Store(1)
y = mk.Store(2)
z = x + y # z is 3, type(z) is int

message = mk.Store("hello")
message = message + " world" 
# message is "hello world", type(message) is str
```
A more detailed breakdown of this behavior is provided in XXXXXXX.

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