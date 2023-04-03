from meerkat.dataframe import DataFrame
from meerkat.engines import TextCompletion
from meerkat.row import Row

import functools
import inspect
from typing import Callable
from meerkat.ops.watch.abstract import WatchLogger


def summary_prompt_one_sentence():
    """A prompt for summarizing text."""
    return """\
Summarize the following text in one sentence.
Text: {text}
Summary:\
"""


def summary_prompt_one_paragraph():
    """A prompt for summarizing text."""
    return """\
Summarize the following text in one paragraph.
Text: {text}
Summary:\
"""

@mk.op()
def summarize(
    df: DataFrame,
    *,
    column: str,
    engine: TextCompletion,
):
    """Summarize every row in a column."""

    def _summarize(row: Row):
        return engine.run(prompt=summary_prompt_one_sentence().format(text=row[column]))

    return df.map(_summarize, inputs="row")



def get_synopsis(title, eral):
    """A prompt for summarizing text."""
    return

def review(synopsis):
    return 

def synopsis_and_review(title, eral):
    synopsis = get_synopsis(title, eral), 
    return review(synopsis), synopsis



@mk.fm
def fn():
    a = engine.run()
    b = engine.run(a)
    return b

# Flow: fn -- input, output, flow
# Task: engine.run -- input, output, flow

"""
# Everything is a flow.

Table: Errands
- code 
- id = (a hash of the code)
- name
- module


Table:  Runs
- id
- engine (optional) (this is an engine, if it exists: base flows will have an engine)
- input_id
- output_id
- parent_id (optional) (this is a parent flow, if it exists)
- errand_id
- latency
- cost $

Table: Engine Runs
- id
- errand_run_id
- engine_id
- prompt
- response
- latency
- cost $
- input_tokens
- output_tokens

Table: Input
- id
- object_id
- errand_id

Table: Output
- id
- object_id
- errand_id

Table: Object
- id
- value
"""

# Fix the model.
# Do it over a dataset containing 10 documents.
# Then I get as output a new set of Tables.
# Errands: 1 row
# Errand Runs: 10 rows
# Input: 10 rows
# Output: 10 rows
# Object: 20 rows

# What can we do with this data?
# 1. Label the quality of the summarizations using
#    (a) human
#    (b) machine, using another errand
#    (rubrics)
# 2a. Embed the data.
# 2. Cluster the data, find patterns yada yada domino etc.

# Now, suppose we vary something: let's vary the engine.
# We get new data.
# Errands: 1 row
# Errand Runs: 20 rows
# Input: 20 rows
# Output: 20 rows
# Object: 40 rows (aim to get 30 objects here)

# What can we do with this data?
# Objective: find out which engine is better where.
# 1. Group by engine, and see how the quality varies (based on some metrics).
# 2. Guided clustering, identify clusters where there are differences between the engines.
# 



def errand(fn: Callable, *, logger: WatchLogger):
    """A decorator for errands."""
    # Assert that one of the arguments of `fn` is `engine`.
    # If not, raise an error.
    signature = inspect.signature(fn)
    if "engine" not in signature.parameters:
        raise ValueError(f"Errands must have an argument called `engine`.")

    # Log the errand.
    errand_id = logger.log_errand(code=inspect.getsource(fn), name=fn.__name__, module=fn.__module__)

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        # Bind the arguments to the function.
        bound = signature.bind(*args, **kwargs)

        # Get the engine.
        engine = bound.arguments["engine"]

        # Collect all the inputs i.e. all arguments except `engine`.
        inputs = {name: value for name, value in bound.arguments.items() if name != "engine"}

        # Log the errand start.
        errand_run_id = logger.log_errand_start(
            errand_id=errand_id,
            inputs=inputs,
            engine=engine.name,
        )

        # Run the errand.
        output = fn(*args, **kwargs)

        # Log the errand end.
        logger.log_errand_end(
            errand_run_id=errand_run_id,
            outputs={"output": output} if not isinstance(output, dict) else output,
        )

        return output

    return wrapper



@mk.errand(logger: Logger=None)
def write_a_poem(
    topic: str,
    engine: TextCompletion = TextCompletion.with_openai(),
):
    """Write a poem."""
    prompt = f"""
    Write a poem about {topic}.
    """
    return engine.run(prompt=prompt)



@mk.fm
def impute()

@mk.fm(engine)
def run_my_chain(engine):
    chain = Chain(engine)
    return chain.run()

chain.run()

@mk.fm()
@jinja_prompt
def poem(topic: str) -> str:
    """Write a poem.
    
    Parameters:
        topic: The topic of the poem.
    """

@mk.fmop
def a():

@mk.fmop
def b():
    a()


b(a())

b()222