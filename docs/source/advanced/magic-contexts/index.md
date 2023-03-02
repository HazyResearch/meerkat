# Magic Contexts

In the {ref}`Store guide <guide_interactive_concepts_store>`, we
briefly mentioned that {class}`~meerkat.Store` objects
can be used with the {py:class}`mk.magic <meerkat.magic>` context.

In this section, we will:

- Explain what the magic context is
- Show how to use the magic context
- Show limitations of the magic context
- Understanding alternatives to the magic context
- Understand best practices for magic

## What is the magic context?

```{important}
Use the {py:class}`mk.magic <meerkat.magic>` context to make methods and attribute access on `Store` objects reactive.
```

The `magic` context is a special context in which methods and attribute access for `Store` objects become reactive. 

This means that you can use `Store` objects in a more natural way, without having to worry about whether or not the statement you're writing is reactive.

## Example
For example, consider an object `Greeting` that has an attribute `name` and a method `get_intro`.

```python
class Greeting:
    def __init__(self, name):
        self.name = name

    def get_intro(self):
        return f"Hello {name}!"
```

Typically, accessing attributes of the wrapped object and calling methods on the wrapped object are not reactive.

```python
greeting = mk.Store(Greeting("Meerkat"))

# Neither of these statements are reactive.
name = greeting.name
intro = greeting.get_intro()

# Even when we indicate the store has updated,
# the statements above are not reactive, so they
# wont update.
greeting.name = "Rhino"
greeting.set(Greeting("Rhino"), triggers=True)
```

With the `magic` context, we can access `name` and call `get_intro` method in a reactive way.

```python
greeting = mk.Store(Greeting("Meerkat"))

with magic():
    name = greeting.name
    intro = greeting.get_intro()

# Now, when we update the store, the statements above
# will update.
greeting.name = "Rhino"
greeting.set(Greeting("Rhino"), triggers=True)

print("name:", name)
print("intro:", intro)
```
