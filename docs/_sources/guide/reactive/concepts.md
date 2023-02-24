(reactivity_concepts)=

# Concepts
In this guide, we'll cover high level concepts related to reactivity.
If you haven't already, please read [](#reactivity) {ref}`Getting Started <reactivity_getting_started>`, which will provide an introduction to reactivity.

(reactivity_concepts_stores)=

## Recap: Stores
Recall, a core principle of reactivity is that Meerkat only tracks `marked` inputs into functions.
If an input is not marked, Meerkat will not track it and will not re-run the function if it changes.

The defacto way to mark an object is to wrap it in a {py:class}`Store <meerkat.Store>`.
In this section, we will briefly cover the importance of `Store`s.
You can read the full guide on `Store`s at [XXXXXXXXXXXX]. 


A `Store` is a special object provided by Meerkat that can be used to wrap arbitrary Python objects, such primitive types (e.g. `int`, `str`, `list`, `dict`), third-party objects (e.g. {py:class}`pandas.DataFrame`, `pathlib.Path`), and even your custom objects. 
A major reason to use `Store` objects is that they make it possible for Meerkat to track changes to Python objects.

Let's look at an example to understand why they are so important to reactive functions. 
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

This is because `x` is just an `int`. By changing `x`, we aren't changing the object that `x` points to (i.e. the `int` `1`). Instead, we are just changing the variable `x` to point to a different object.

*What we need here is to update the object that `x` points to.* It's impossible to do this with a regular `int`. We can do this with a `Store` instead.

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store

x.set(4, triggers=True)
print(z)
# z is now Store(6), type(z) is Store
```
By calling `set()` on `x`, we are changing the object that `x` points to. This is what allows `z` to be updated.
Ignore the `triggers=True` argument for now, we discuss it in more detail in {ref}`the section below <reactivity_concepts_reactive_functions>`.)

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

(reactivity_concepts_reactive_functions)=

## Demystifying Reactive Functions

Now that we understand the importance of `Store` objects to reactivity, let's look at how reactive functions work.
Consider the same function `add`.

**What happens when you wrap `add` with `mk.reactive()`?**
    
```python
@mk.reactive()
def add(a, b):
    return a + b
```

Meerkat tracks the function `add` and re-runs it if any of the inputs to it (i.e. `a` or `b`) change.
Here, we are assuming both `a` and `b` are {ref}`marked <reactivity_concepts_markables>`.
Internally, Meerkat includes `add` in a computation graph that tracks dependencies between reactive functions.

**What happens when you call `add`?**
```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store
```

Meerkat does three things:
1. **Unwrap input stores:** Any inputs to `add` that are `Store` objects before passing them into the body of `add`.
1. **Turn off reactivity:** Meerkat turns off reactivity only for the duration of the function call.
1. **Wrap and mark the outputs**: The output of `add` is wrapped in a `Store` object and returned. If the output is a Markable object (e.g. `mk.DataFrame <meerkat.DataFrame>`), the object is marked. This lets you use the output of `add` as input to other reactive functions.
1. **Turn on reactivity:** Meerkat turns reactivity back on once the function has returned.

The main consequence of these steps is that **any reactive function can be written as a regular Python function**.
You don't have to worry about if the inputs are stores or if you have to manage reactivity manually.
This will allow you to convert any existing Python function into a reactive function simply by wrapping it in `mk.reactive()`.

**What happens when you change any of the inputs to `add`?**

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y) # z is Store(3), type(z) is Store

x.set(4, triggers=True)
# z is now Store(6), type(z) is Store
```

```{note}
`triggers=True` is an optional argument to `Store.set()` to trigger the reactive chain.
It should never be called explicitly in the code, but can be helpful for debugging purposes.
```

What does Meerkat do when `x` is changed? Let's walk through the steps:
- Meerkat detects that `x` changed because `x` is a `Store`, and `.set()` was called on it.
- Meerkat looks for all reactive functions that were called with `x` as an argument (i.e. `add`).
- Meerkat then re-runs those functions (i.e. `add`) and any functions that depend on their outputs.
- Finally, Meerkat updates the outputs of any reactive functions that it re-runs (i.e. `z`).


(reactivity_concepts_markables)=

## Markables and Reactivity
Earlier in this section, we mentioned the importance of marking inputs to reactive functions.
Let's break down the principle behind marked objects and how unmarking them can actually be useful at times.

To recap, {py:class}`Store <meerkat.Store>`, {py:class}`mk.DataFrame <meerkat.DataFrame>`, and {py:class}`Column <meerkat.DataFrame>` objects are all Markable objects.
In order for a reactive function to re-run when an input changes, the input must be marked.
We can toggle the mark on an object by calling `mark()` or `unmark()`.

```python
# Stores are marked by default.
x = mk.Store(1)

df = mk.DataFrame({"a": [1, 2, 3]}).mark()
col = mk.ScalarColumn([1, 2, 3]).mark()
```

### Understanding the `marked` state

```{important}
The marked state of an input controls whether or not the function will be re-run when the value of the input changes.
```

Marking an input indicates to a reactive function that changes to the input should trigger the function to re-run.
If an input is unmarked or unable to be marked, the function will not be triggered to re-run when the input changes.

Note that it is at the time the function is called, that an input's marked state determines if that function will retrigger.

**What happens when we unmark an object after passing it to a reactive function?**

Nothing!
If the input was marked at the time the reactive function was run, the function will react to changes in that input, even if the input is subsequently unmarked.
The marked state is only read when the function is called.

Consider the case where we unmark `a` after passing it to `add`.

```python
@mk.gui.reactive()
def add(a, b):
    return a + b

a = mk.Store(1)
b = mk.Store(2)
c = add(a, b) # c is Store(3), type(c) is Store

a.unmark()
d = add(a, b)
# d only updates when b changes, because a was unmarked when passed to the function that returned d.
# However, c will still update when a changes. This is because a was marked when passed to the function that returned c.
```

### Unmarking inputs to reactive functions
It seems natural to only want to pass marked inputs to reactive functions.
This way we can be sure that the function will re-run when any of its input changes.

```{important}
Unmarked inputs to reactive functions will not trigger the function to re-run.
However, if the function is re-run, the newest value of the unmarked input will be used.
```

However, in some cases, we may want to be selective about which inputs should trigger the function.
Perhaps, we only want to trigger the function when a certain input changes.

To achieve this, we can simply unmark the inputs that we don't want to trigger the function.
For example, say we do not want `add` to re-run when `a` changes, but we do want it to re-run when `b` changes.

```python
@mk.gui.reactive()
def add(a, b):
    return a + b

a = mk.Store(1).unmark()
b = mk.Store(2)
c = add(a, b)

a.set(4, triggers=True)
# c is still Store(3) because a was unmarked when passed to the function that returned c.
# Thus changes in a will not trigger the function to re-run.
```

### Non-markable inputs to reactive functions
We may not always pass markable objects to reactive functions.
For example, I may pass a list to a function.

If a object that is not markable is passed to a reactive function, the function will not re-run when the object changes.
However, if the function is ever retriggered (e.g. by changing another marked input into the function), the newest value of the unmarked input will be used.
**This only works when the object is modified in-place** - i.e. the input needs to be mutable.
To understand why this happens, see the discussion on {ref}`pass-by-assignment <reactive_concepts_pass_by_reference>`.

A great example of this is a `list`. Below, we define a function that takes a list and a value, and returns the sum of the list and the value. Note that a list, when it is not wrapped in a `Store`, is not a markable object.

```python
def sum_list_with_value(my_list: list, x: int):
    return sum(my_list) + value

my_list = [1, 2, 3]
x = mk.Store(4)
out = sum_list_with_value(my_list, x)
print("out", type(out), out)
# out is Store(10)

my_list.append(4)  # my_list = [1, 2, 3, 4]
print("out", type(out), out)
# out is still Store(10) because the function was not re-run when my_list changed.

x.set(5, triggers=True)
# The value of x changed with `triggers=True`, so the function was re-run.
print("out", type(out), out)
# out is now Store(15) - sum([1, 2, 3, 4]) + 5
```

(reactive_concepts_pass_by_reference)=

### [Aside] Pass by Reference vs Pass by Value
Recall, Python passes arguments by assignment.
If you are not familiar with passing by assignment, we can very loosely boil it down to this:
- immutable objects (e.g. primitives - `int`, `float`, `str`, `bool`, etc.) are passed by value
- mutable objects (e.g. `list`) are passed by reference

*NOTE*: This is not a perfect analogy, but it is sufficient for our purposes.

Reactive functions are just like regular Python functions in this way.
Immutable inputs into a reactive function can never change. If the reactive function is re-run, the input will be the same as it was the first time.
In contrast, mutable inputs can change. If the input was modified in-place and the reactive function is re-run, the input will be different than it was the first time.

This does not mean that if any mutable inputs are changed, the reactive function will re-run.
The mutable input must both be `marked` and be changed in a special way `.set()` to trigger
the reactive function to re-run. We will see why this is important in the next section.

Any markable object in Meerkat is a mutable object.
This means there are operations that can be performed on that object in-place.
For example, we can set the value of a `Store` using `.set()`.

```python
a = mk.Store(1)
print(id(a))

a.set(2)  # modifies the store `a` in-place
print(id(a))
```


## Reactivity as a Graph

We can think of reactivity as a graph of inputs, functions, and outputs.
In fact, Meerkat maintains a graph of reactive functions and their inputs and outputs to determine what functions should be re-run.

**What are the nodes?**

Reactive functions and their inputs and outputs are all nodes in the graph.

When a reactive function is called with inputs, a node is created for the function call and its inputs.
The node for the function call is referred to as an `Operation`.
The operation corresponds to this particular function call. If the function is called again, a new operation is created.


**What are edges in this graph?**
Edges indicate what variables are inputs to a function, and what variables are outputs of a function.
These edges are directed - i.e. input -> operation -> output.

There are two kinds of edges in the graph:
1. **Trigger Edges**: These edges indicate which variables should trigger an operation to re-run.
2. **Connector Edges**: These edges simply indicate that variables are inputs/outputs (i.e. connected) to an operation. Connector edges between an input and an operation indicate that changes in the input will not trigger the operation to re-run.

Trigger edges are used to describe relationships between inputs and the operation. In other words,
an input *triggers* an operation to re-run if the input points to the operation with a trigger edge.

This distinction is important because it allows us to be selective about which inputs should trigger a function to re-run.
For example, `c = add(a,b)` only re-runs when `b` changes.

```python
@mk.reactive()
def add(a, b):
    return a + b

a = mk.Store(1).unmark()
b = mk.Store(2)
c = add(a, b)
```

The graph looks like this:

```{figure} ../../../assets/guide/graph/add.png
```

TODOS:

- What happens when reactive function doesnt have arguments
- Using the same reactive function multiple times