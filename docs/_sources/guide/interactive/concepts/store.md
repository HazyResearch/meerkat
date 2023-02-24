(guide_interactive_concepts_store)=

# Stores

An interactive application needs some way of creating variables that capture
the state of the application. This is important to 

- keep track of application state that is going to change over time
- keep this state in sync between the frontend and Python code
- provide explicit ways to manipulate this state
- implement and debug the application in terms of this state

In Meerkat, wrapping a Python object in a `Store` object provides a way to do this. These pages will explain:
- what a `Store` is
- how to work with `Store` objects
- how `Store` objects are used in interactive applications
- common pitfalls and how to avoid them


## What is a `Store`?

The `Store` is a wrapper that can be used to wrap an arbitrary Python object. For example, a `Store` can be used to wrap an `int`.

```python
x = mk.Store(1)
```
A `Store` is itself a Python object. For instance, calling `type(x)` will return `Store`.
```python
type(x) # Store
```

**`Store` objects are designed to be as transparent as possible, so that they behave like the object they wrap.** For example, you can use `Store` objects in the same way you would use the object they wrap.
```python
x + 1 # 2
assert isinstance(x, int) # this works!
```

**However, while the way in which you use the `Store` is exactly the same, the consequences of using it are different.** We've distilled it down to the following rules.

**When do `Store` objects behave exactly like the object they wrap?**
`Store` objects behave exactly like the object they wrap when you call their methods i.e. you use `.method()` syntax.
```python
x = mk.Store('HELLO WORLD')
y = x.lower() # 'hello world'
type(y) # str
```

`Store` objects behave exactly like the object they wrap when you access their attributes i.e. you use `.attribute` syntax.
```python
x = mk.Store('HELLO WORLD')
y = x.lower # <built-in method lower of str object at ...>
type(y) # <class 'builtin_function_or_method'>
```
This is quite useful when wrapping complex objects.
```python
import pandas as pd
x = mk.Store(pd.DataFrame({'a': [1, 2, 3]}))
y = x.columns
type(y) # <class 'pandas.core.indexes.base.Index'>
```

**When do `Store` objects behave almost exactly like the object they wrap?**
`Store` objects behave (almost) exactly like the object they wrap when you use them as inputs to functions. For instance, the following code works as expected.
```python
x = mk.Store('HELLO WORLD')
y = len(x) # 11
type(y) # int
```
However, sometimes `Store` objects may behave unexpectedly when used with arbitrary functions. This mostly happens when functions are not duck-typed, but instead have behavior that expects specific types. It can also happen when a function is actually implemented in C.

Here's a simple example that fails from Python's `os` module.
```python
x = mk.Store('./relative/path/to/file')
y = os.path.abspath(x)
# TypeError: expected str, bytes or os.PathLike object, not Store
```
The workaround is simple. If you encounter an error of this kind, use the `.value` attribute to get the underlying object and pass that in to the function instead.
```python
x = mk.Store('./relative/path/to/file')
y = os.path.abspath(x.value)
type(y) # str
```


#### When do `Store` objects behave differently from the object they wrap?
By design, there are important situations in which `Store` objects behave differently from the objects they wrap. Our goal is make these situations as salient as possible so that you can understand why they behave differently, and how to work with them correctly.

_If you notice any situations in which `Store` objects behave differently from the objects they wrap that we have not documented here, please let us know by filing an issue on [GitHub](https://github.com/hazyresearch/meerkat)!_

**1. When you use them as inputs to reactive functions.**

Let's start with the most important difference between `Store` objects and the objects they wrap. This is their behavior when you use them as inputs to reactive functions. 

To recap, reactive functions are functions that are automatically re-executed when their inputs change. `Store` objects are one of the object types that when passed as inputs to a reactive function, will trigger the function to re-execute when the value of the `Store` changes.

However, if the underlying object was used directly as an input without being wrapped in a `Store`, the function would not be re-executed when it changes. This is because the underlying object is not a `Store`, and Meerkat does not know that it should re-execute the function when it changes.

**2. With operators.**

There are some situations where it desirable that `Store` objects actually behave differently from the objects they wrap. 

For example, it is useful to be able to add two `Store` objects together, and have the result be a `Store` object. In addition, it is particularly convenient if this calculation is reactive, so that the result is automatically updated when either of the inputs change.

```python
x = mk.Store(1)
y = mk.Store(2)
z = x + y
type(z) # Store
# z is automatically updated when x or y changes

w = x + 1
type(w) # Store
# w is automatically updated when x changes
```

**The rule we use to determine whether an operation on a `Store` should be reactive is simple: if the operation is written with only symbols and no letters, numbers or parentheses, it will be reactive.**

This means that `+`, `-`, `*`, `/`, `**`, `//`, `%`, `<<`, `>>`, `&`, `|`, `^`, `~`, `==`, `!=`, `<`, `<=`, `>`, `>=` will all be reactive when used with atleast one `Store` object. 
Their shortcut counterparts, `+=`, `-=`, `*=`, `/=`, `**=`, `//=`, `%=`, `<<=`, `>>=`, `&=`, `|=`, `^=` will also be reactive.

However, all of the following ways of interacting with `Store` objects will **not** behave reactively. These all contain letters, numbers or parentheses, and so will not be reactive.
- `is`, `is not`, `in`, `not in`, `and`, `or`, `not`. For example, `x is y` will not be reactive, even if `x` and `y` are `Store` objects. As alternatives to these, we provide the `mk.is`, `mk.is_not`, `mk.in`, `mk.not_in`, `mk.and`, `mk.or`, and `mk.not` functions, which are reactive.
- `f-string` formatting, such as `f'{x}'`. For these, we recommend either using the `+` operator with `str` objects and `Store` objects that wrap strings, or creating a reactive function that returns the formatted string. For example, `mk.reactive(lambda x: f'{x}')(x)` will be reactive.
- No method calls, such as `x.lower()` will be reactive. The best way to call methods as reactive functions is to wrap them in with `mk.reactive` before calling them. For example, `mk.reactive(x.lower)()` will be reactive.
- Attribute access, such as `df.columns` will not be reactive. The best way to access attributes as reactive functions is to create a `lambda` function wrapped in `mk.reactive`. For example, `mk.reactive(lambda x: x.columns)(df)` will be reactive.
- Function calls of the form `f(x)` will not be reactive, unless `f` is decorated with `@mk.reactive`.
- Calling `Store` objects will not be reactive. For example, `x()` will not be reactive. The best way to call `Store` objects as reactive functions is to just create a reactive function that calls them. For example, `mk.reactive(lambda x: x())(x)` will be reactive.
- Indexing and slicing, such as `x[0]` or `x[0:5]` will not be reactive. The best way to index and slice `Store` objects is to create a `lambda` function wrapped in `mk.reactive`. For example, `mk.reactive(lambda x: x[0])(x)` will be reactive.

While this rule might be silly, we designed it to be as simple as possible to remember, while making `Store` objects convenient to use. If you can remember that `Store` objects behave reactively when you use them with operators, you will be able to use them correctly in all situations.

One situation we want to call out in particular is the use of common Python built-in functions on `Store` objects. For example, type casting like `int(...)` or `str(...)` will not only not be reactive, but will also return `int` and `str` objects respectively, and not `Store` objects. This can be surprising and lead to strange situations where the "chain of reactivity" is broken, but we believe this is the correct behavior to ensure that `Store` objects behave as expected. 

As a convenience, we provide the `mk.int`, `mk.float`, `mk.str`, `mk.bool`, `mk.list`, `mk.tuple`, `mk.dict`, `mk.set` functions, which are all reactive functions that return `Store` objects. A full list of reactive functions provided by Meerkat can be found [XXXXXXXXXXXXX](XXXXX).

## Common Gotchas with `Store` objects




**When should I use `Store` objects?**
`Store` objects are central to Meerkat's interactive functionality, and serve several important roles when building an application with Meerkat.

That said, when writing "normal" Python code in a Meerkat application, do not use `Store` objects unless you need them.

There are several reasons to use `Store` in Meerkat.

**1. To create a variable that can be modified by the frontend or Python code.**

Normally, Python variables cannot be manipulated by the frontend directly. Wrapping a Python variable in a `Store` allows it to be synchronized with the frontend.
```python
number = mk.Store(0)
slider = mk.gui.Slider(value=number)
```
In fact, all Meerkat components like `Slider` will automatically wrap their inputs in a `Store` (if required) when they are initialized. This ensures that their attributes are always synchronized with the frontend, and you can use them knowing that they will always be up-to-date.

**2. To create a variable that can trigger re-execution of a reactive function.**

Reactive functions are functions that are automatically re-executed when their inputs change. `Store` objects are one of the object types that when passed as inputs to a reactive function, will trigger the function to re-execute when the value of the `Store` changes.

For example, the following code will print the updated value of `number` every time the slider is moved.
```python
@mk.reactive()
def print_number(number):
    print(number)

number = mk.Store(0)
slider = mk.gui.Slider(value=number)

print_number(number)
```

While this is an example where the fact that a `Store` is automatically synchronized with the frontend triggers re-execution of a reactive function, it is not the only way to trigger re-execution.

This can also be done by using the `.set` method on a `Store` inside an endpoint. In the example below, the `print_number` function will be re-executed every time the button is clicked, since the `set_number` endpoint will update the value of `number`.

```python
@mk.endpoint()
def set_number(number: Store):
    number.set(number + 1)

@mk.reactive()
def print_number(number):
    print(number)

number = mk.Store(0)
button = mk.gui.Button(title="Click me!", on_click=set_number.partial(number=number))

print_number(number)
```


## FAQs
#### How do I create a `Store`?
```python
x = mk.Store(1) # or mk.Store(<any object>)
```

#### What can I put in a `Store`?
_Anything._ All Python objects are valid inputs to `Store`. For example, primitive types (e.g. `int`, `float`, `str`, etc.), complex types (e.g. `list`, `dict`, `tuple`, etc.), arbitrary objects (e.g. `pandas.DataFrame`, `numpy.ndarray`, etc.), classes, functions and even other `Store` objects (although we don't recommend nesting `Store` objects).

#### Will this `<statement>` be reactive, and return a `Store`?
Normally, statements with `Store` objects are reactive *only* when they involve operators. For example, `x + y`, `x - 1` and `~x` will all behave as reactive statements that return another `Store` object.

The only other situation in which this occurs is when a reactive function is being called on a `Store` object. For example, `mk.reactive(lambda x: x + 1)(x)` will be reactive, and return a `Store` object.

Finally, many other statements on `Store` objects will become reactive in the `magic` context. Read more about the `magic` context [XXXXXXXXX](XXXXX).

#### How do I update a `Store`?

A `Store` can be updated by either the Python code and/or frontend.

- *Frontend*: If the store is used by a component on the frontend, changes on the frontend will automatically be reflected in the `Store` on the Python side.
- *Python*: The `Store` has a `.set` method that can be used to update its value. This `.set` method should only be used in functions decorated with `@mk.endpoint` if you want the updated `Store` to trigger re-execution of reactive functions.

```python
counter = mk.Store(0)
counter.set(1)
```

Remember, setting a store equal to a value (i.e. ``store = 1``) will not update the value of the store. Use ``store.set()``!

#### Why does `Store(None) is None` return `False`?

Unfortunately, `is` is a special token in Python that cannot be intercepted by Meerkat, so this is an unavoidable limitation of the `Store` object. However, `Store(None) == None` will return `True`.

```python
Store(None) is None  # False
Store(None) == None  # True
```

#### Do I need to do anything special inside reactive functions?
No. Write your reactive functions as you would any other Python function.

Reactive functions unwrap all `Store` arguments automatically, so treat those arguments as normal Python objects. They also automatically wrap the return value of the function in a `Store` object, so return any Python object you like.
```python
@mk.reactive()
def add(x: int, y: int):
    # Don't worry about `Store` objects here!
    # If x and y are passed in as Store objects, 
    # they've already been unwrapped to their 
    # wrapped values automatically.
    return x + y # this is just adding two `int`s

x = mk.Store(1)
y = mk.Store(2)
add(x, y) # returns a Store containing the value 3
```
Of course, you can still create `Store` objects manually inside reactive functions if you want to.

#### Should I use the `magic` context?
It depends. 

The `magic` context is a special context in which all methods and attribute access for `Store` objects become reactive i.e. `store.<blah>`-like statements (remember that `fn(store)` calls will not be reactive unless `fn` is explicitly decorated with `mk.reactive()` though!). 

This means that you can use `Store` objects in a more natural way, without having to worry about whether or not the statement you're writing is reactive.

For example, instead of writing:
```python
df = mk.Store(pd.DataFrame({"a": [1, 2, 3]}))
# This doesn't work like a reactive statement
cols = df.columns
# You must wrap `df.columns` in a reactive function
# if you want the `cols` variable to be updated when `df` changes.
cols = mk.reactive(lambda df: df.columns)(df)
```
You can write:
```python
df = mk.Store(pd.DataFrame({"a": [1, 2, 3]}))
# No need to wrap `df.columns` in a reactive function
# if you're using the `magic` context.
with mk.magic():
    cols = df.columns
```
The `magic` context is not required at all, but it can make your code more succinct and easier to write. In general, we recommend using the `magic` context as little as possible, since it will make it much less explicit to a reader of your code whether or not a statement is reactive. Anything you can do in the `magic` context, you can also do without it.

#### Are there any caveats to using the `magic` context?
There are some important edge cases when using the `magic` context that you should be aware of. You should see warnings when you encounter these edge cases, but it's still good to be aware of them.

One major edge case is when using `and`, `or`, `not`, `is`, and `in` operators. These operators are special tokens in Python, and cannot be intercepted by Meerkat. This means that the following statements will not be reactive, and should not be relied on.
```python
x = mk.Store(1)
y = mk.Store(2)
w = mk.Store([1, 2, 3])
with mk.magic():
    # None of these will be reactive
    # and may return the wrong value!
    z = x and y
    z = x is y
    z = x or y
    z = not x
    z = x in w
    # Use Meerkat's built-in overloads instead
    mk.cand(x, y)
    mk.cor(x, y)
    mk.cnot(x)
    # Or just wrap in a reactive lambda function!
    z = mk.reactive(lambda x, y: x and y)(x, y)
```

#### What's a quick way to debug the value of `Store` objects?
The easiest way is to use the `mk.print` function. This is a reactive `print` function that will re-print the value of a `Store` object whenever it changes.
```python
x = mk.Store(1)
mk.print(x) # prints 1
# Will re-print the value of `x` whenever it changes
```
Underneath the hood, `mk.print` is just `mk.reactive(print)`, so you can also use it to print the value of any Python object.

You can follow this pattern to create more advanced debugging tools.


#### Why do I get different `Store` objects when I index into a `Store` repeatedly?

This is a limitation of the `Store` object. When you index into a `Store` object (e.g. `store[0]`), you get a new `Store` object that wraps the value at that index. This means that if you index into a `Store` object multiple times, you will get different `Store` objects each time.

```python
x = mk.Store([1, 2, 3])
s1 = x[0]
s2 = x[0]
s1 is s2   # This is False

x = mk.Store({"a": 1, "b": 2})
s1 = x["a"]
s2 = x["a"]
s1 is s2   # This is False
```


## Faux Pas

- A list of Stores is not reactive, a Store of list is reactive

```python
    
    # my_list is a list of stores. It is not a Store.
    # Operations on my_list will not trigger reactions.
    my_list = [mk.gui.Store(0), mk.gui.Store(0)]
    my_list += [mk.gui.Store(2))]  # this does nothing

    # my_stores is a store containing list of integers.
    # appending will change the value of my_stores.
    # This will trigger reactions.
    my_stores = mk.gui.Store([0, 1])
    my_stores += [2]
```

- Using shortcut operators (``and``, ``or``, ``not``) with Stores will not return Stores, but using Meerkat's built-in overloads (``mk.cand``, ``mk.cor``, ``mk.cnot``) will

```python

    store = Store("")
    # These will not return Stores
    type(store or "default")  # str
    type(store and "default")  # str
    type(not store)  # bool

    # These will return Stores
    type(mk.cor(store, "default"))  # Store
    type(mk.cand(store, "default"))  # Store
    type(mk.cnot(store))  # Store

```
