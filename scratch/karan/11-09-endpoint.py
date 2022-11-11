from typing import Dict, List

from pydantic import BaseModel

import meerkat as mk
from meerkat.interactive.graph import endpoint

counter = mk.gui.Store(0)


def _increment(count: int, step: int = 1) -> int:
    return count + step


@endpoint
def increment(count: int, step: int = 1) -> int:
    count = _increment(count, step) # magic step!

# Less magic option
# Disadvantage: can't pass either Store or int to the `count` argument
# must pass only Store
# @endpoint
# def increment(count: mk.gui.Store, step: int = 1) -> int:
#     count.value = _increment(count.value, step)


increment(counter, 1)
increment(counter, 1)
increment(counter, 1)
print("Count:", counter.value)
print("Done")

my_increment = endpoint(_increment, return_into=counter)
my_increment(counter)
print(counter.value)


@endpoint
def increment_many(counts: List[int], step: int = 1) -> List[int]:
    for i, count in enumerate(counts):
        counts[i] = _increment(count, step) # list update

some_counters = [mk.gui.Store(-1), mk.gui.Store(0), mk.gui.Store(1)]
increment_many(some_counters, 1)
print([counter.value for counter in some_counters])


@endpoint
def increment_many(counts: List[int], step: int = 1) -> List[int]:
    counts[1] = _increment(counts[1], step) # list update

some_counters = [mk.gui.Store(-1), mk.gui.Store(0), mk.gui.Store(1)]
increment_many(some_counters, 1)
print([counter.value for counter in some_counters])

@endpoint
def increment_many(counts: Dict[str, int], step: int = 1) -> List[int]:
    for key, count in counts.items():
        counts[key] += step # dict update, with += assignment

some_counters = {"a": mk.gui.Store(-1), "b": mk.gui.Store(0), "c": mk.gui.Store(1)}
increment_many(some_counters, 1)
print({key: counter.value for key, counter in some_counters.items()})

@endpoint
def increment_many(counts: list, step: int = 1) -> List[int]:
    for i, count in enumerate(counts):
        counts[i] = _increment(count, step)

some_counters = mk.gui.Store([-1, 0, 1])
increment_many(some_counters, 1)
print(some_counters.value)
print("---")

@endpoint
def increment_many(counts: list, step: int = 1) -> List[int]:
    new_counts = []
    for i, count in enumerate(counts):
        new_counts.append(_increment(count, step))
    counts = tuple(new_counts)

some_counters = mk.gui.Store(tuple([-1, 0, 1]))
increment_many(some_counters, 1)
print(some_counters.value)
print("---")



class Counter(BaseModel):
    count: int = 0

    # Also fine if we don't use Pydantic
    # def __init__(self, count: int = 0):
    #     self.count = count

    def json(self):
        return {"count": self.count}

    def increment(self, step: int = 1):
        self.count += step

    def __eq__(self, other):
        return self.count == other.count


@endpoint
def increment_many_objs(counts: Dict[str, Counter], step: int = 1) -> List[int]:
    for key, count in counts.items():
        count.increment(step) # this won't work!!

some_counters = {"a": mk.gui.Store(Counter(count=-1)), "b": mk.gui.Store(Counter(count=0)), "c": mk.gui.Store(Counter(count=1))}
increment_many_objs(some_counters, 1)
print({key: counter.value.count for key, counter in some_counters.items()})
print("^ Even though the objects were updated, no Store Modifications were detected!")

@endpoint
def increment_many_objs(counts: Dict[str, Counter], step: int = 1) -> List[int]:
    for key, count in counts.items():
        count.count += step # this won't work!!

some_counters = {"a": mk.gui.Store(Counter(count=-1)), "b": mk.gui.Store(Counter(count=0)), "c": mk.gui.Store(Counter(count=1))}
increment_many_objs(some_counters, 1)
print({key: counter.value.count for key, counter in some_counters.items()})
print("^ Even though the objects were updated, no Store Modifications were detected!")

@endpoint
def increment_many_objs(counts: Dict[str, Counter], step: int = 1) -> List[int]:
    for key, count in counts.items():
        count.count += step # this won't work!!
    counts = counts # but this does as the Store variable appears on the left side of an assignment

some_counters = {"a": mk.gui.Store(Counter(count=-1)), "b": mk.gui.Store(Counter(count=0)), "c": mk.gui.Store(Counter(count=1))}
increment_many_objs(some_counters, 1)
print({key: counter.value.count for key, counter in some_counters.items()})

"""

import ast
import inspect
class CustomNodeVisitor(ast.NodeVisitor):
    def __init__(self): 
        self.targets = set()

    def visit_Assign(self, node): 
        # Extract all the ids from the targets
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.targets.add(target.id)
            elif isinstance(target, ast.Attribute):
                self.targets.add(target.value.id)
            elif isinstance(target, ast.Subscript):
                self.targets.add(target.value.id)
            elif isinstance(target, ast.Tuple):
                for elt in target.elts:
                    self.targets.add(elt.id)
            elif isinstance(target, ast.List):
                for elt in target.elts:
                    self.targets.add(elt.id)
            else:
                raise ValueError("Unknown target type: ", type(target))

    visit_AnnAssign = visit_Assign
    
    def visit_AugAssign(self, node):
        if isinstance(node.target, ast.Name):
            self.targets.add(node.target.id)
        elif isinstance(node.target, ast.Attribute):
            self.targets.add(node.target.value.id)
        elif isinstance(node.target, ast.Subscript):
            self.targets.add(node.target.value.id)
        else:
            raise ValueError("Unknown target type: ", type(node.target))
    
    def finalize(self):
        if "_" in self.targets:
            self.targets.remove("_")
        if "self" in self.targets:
            self.targets.remove("self")
        return self.targets


import sys
def call_function_get_frame(func, *args, **kwargs):
    """https://stackoverflow.com/questions/4214936/how-can-i-get-the-values-of-
    the-locals-of-a-function-after-it-has-been-executed Calls the function.

    *func* with the specified arguments and keyword arguments and snatches its
    local frame before it actually executes.
    """

    frame = None
    trace = sys.gettrace()

    def snatch_locals(_frame, name, arg):
        nonlocal frame
        if frame is None and name == "call":
            frame = _frame
            sys.settrace(trace)
        return trace

    sys.settrace(snatch_locals)
    try:
        result = func(*args, **kwargs)
    finally:
        sys.settrace(trace)
    return frame, result

def endpoint(
    fn: Callable = None,
):
    if fn is None:
        # need to make passing args to the args optional
        # note: all of the args passed to the decorator MUST be optional
        return partial(
            endpoint,
        )

    # Parse the function: get the names of targets that are assigned to
    # in the function
    fn_ast = ast.parse(inspect.getsource(fn))
    visitor = CustomNodeVisitor()
    visitor.visit(fn_ast)
    targets = visitor.finalize()

    # Analyze the fn signature
    fn_signature = inspect.signature(fn)

    def _endpoint(fn: Callable):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            """
            This `wrapper` function is only run once. It creates a node in the
            operation graph and returns a `Box` object that wraps the output of the
            function.

            Subsequent calls to the function will be handled by the graph.
            """
            # TODO(Sabri): this should be nested
            # Unpack the args and kwargs from the boxes and stores
            unpacked_args, unpacked_kwargs, _, _ = _unpack_boxes_and_stores(
                *args, **kwargs
            )

            # Now, we need to figure out which boxes and stores were updated
            # and create Modifications for them
            # visitor.targets contains strings of the names of the variables
            nonlocal fn_signature
            fn_bound_arguments = fn_signature.bind(*args, **kwargs).arguments

            # Run through the targets and check which ones are boxes and stores
            # in the function signature
            modified = {}
            for target in targets:
                if target in fn_bound_arguments:
                    arg = fn_bound_arguments[target]
                    if _has_box_or_store(arg):
                        modified[target] = arg

            # fn will update some of the boxes and/or stores
            frame, result = call_function_get_frame(fn, *unpacked_args, **unpacked_kwargs)

            # Inspect the local frame of the build function
            update = {}
            for key in modified:
                update[key] = frame.f_locals[key]

            # print("update", update)
            # print("modified", modified)
            # print("result", result)
            # print("targets", targets)

            # Update the boxes and stores and return modifications
            modifications = []
            _update_result(
                result=modified,
                update=update,
                modifications=modifications,
            )
            
            # Questions:
            # What to do with the result? Need to send it to the frontend
            # Need to construct Modifictations for all changes

            if return_into is not None:
                # Stateless function, so return values
                # need to be used to update the boxes and stores
                _update_result(result=return_into, update=result, modifications=modifications)
            else:
                # Function with side effects, so return values
                # go straight to the frontend
                pass

            print(modifications)
            return result

        return wrapper

    return _endpoint(fn)
"""