# Creating Reactive Functions

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

We'll look at this in more detail in a bit. First, though, we'll talk about `Markable` objects, which are intertwined with reactive functions.

(reactivity_concepts_reactive_functions)=

## Demystifying Reactive Functions

Now that we understand the importance of `Store` objects to reactivity, let's look at how reactive functions work. Consider the same function `add`.

**What happens when you wrap `add` with `mk.reactive()`?**

```python
@mk.reactive()
def add(a, b):
    return a + b
```

Meerkat tracks the function `add` and reruns it if any of the inputs to it (i.e., `a` or `b`) change. Here, we are assuming that both `a` and `b` are {ref}`marked <reactivity_concepts_markables>`. Internally, Meerkat puts the `add` function into a computation graph that tracks dependencies between reactive functions.

**What happens when you call `add`?**

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y)  # z is Store(3), type(z) is Store
```

Meerkat does the following steps:

1. **Unwrap input stores.** Any inputs to `add` that are `Store` objects are unwrapped before passing them into the body of `add`.
2. **Turn off reactivity.** Meerkat turns off reactivity before executing the function. This only applies for the duration of the function call.
3. **Execute the function.** The function is executed as if it were a normal Python function.
4. **Wrap and mark the outputs**: Then, the output of `add` is wrapped in a `Store` object and returned. If the output is a markable object (e.g., `mk.DataFrame <meerkat.DataFrame>`), the object is marked. This lets us use the output of `add` as input to other reactive functions.
5. **Turn on reactivity:** Meerkat turns reactivity back on once the function has returned.

The main consequence of these steps is that **any reactive function can be written as a regular Python function**. We don't have to worry about unwrapping `Store` inputs or managing reactivity manually. Thus, we can convert any existing Python function into a reactive function simply by wrapping it with `mk.reactive()`.

**What happens when you change any of the inputs to `add`?**

```python
x = mk.Store(1)
y = mk.Store(2)
z = add(x, y)  # z is Store(3), type(z) is Store

x.set(4, triggers=True)
# z is now Store(6), type(z) is Store
```

What does Meerkat do when `x` is changed? Let's walk through the steps:

- Meerkat detects that `x` changed because `x` is a `Store`, and `.set()` was called on it.
- Meerkat looks for all reactive functions that were called with `x` as an argument (i.e. `add`).
- Meerkat then reruns those functions (i.e., `add`) and any functions that depend on their outputs.
- Finally, Meerkat updates the outputs of any reactive functions that it reruns (i.e. `z`).

TODOS:

- What happens when reactive function doesnt have arguments
- Using the same reactive function multiple times
