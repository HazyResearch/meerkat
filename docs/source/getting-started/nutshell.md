Robustness Gym in a Nutshell
================================

What is Robustness Gym? Should you use it? Read this page to find some quick answers
 to common questions.


The Big Picture
--------------------

Robustness Gym was built out of our own frustrations of being unable to systematically
 evaluate and test our machine learning models. 
 
 Traditionally, evaluation has consisted of a few simple steps:
  1. Load some data
  2. Generate predictions using a model
  3. Compute aggregate metrics
  
 This is no longer sufficient: models are increasingly being deployed in real-world use 
 cases, and aggregate performance is too coarse to make meaningful model assessments. 
 Modern evaluation is about understanding if models are _robust_ to all the
  scenarios they might encounter, and where the tradeoffs lie.
 
 This is reflected in Robustness Gym which distills these modern goals
  into a new workflow,
 1. Load some data
 2. Compute and cache side-information on data 
 3. Build slices of data
 4. Evaluate across the slices
 5. Report and share findings
 6. Iterate

We'll go into what these steps mean and how to use them in Robustness Gym next.

The Robustness Gym Workflow
----------------------------

### 1. Load some data
Loading data in Robustness Gym is easy. We extend the Huggingface 
[datasets](https://github.com/huggingface/datasets) library, 
so all datasets there are immediately available for use using the Robustness Gym
 `Dataset` class.

```python
import robustnessgym as rg 

# Load the boolq data
dataset = rg.Dataset.load_dataset('boolq', split='train[:10]')

# Load the first 10 training examples
dataset = rg.Dataset.load_dataset('boolq', split='train[:10]')

# Load from jsonl file
dataset = rg.Dataset.from_json("file.jsonl")
```

### 2. Compute and cache side-information

One of the most common operations in evaluation is interpreting and analyzing
 examples in dataset. 
This can mean tagging data, adding additional information about examples from a
 knowledge base, or making predictions about the example.
 
 It's often useful to have this information available conveniently stored alongside
  the example, ready to use for analysis.
  
  This is the idea of the `CachedOperation` class in Robustness Gym. Think of it as a
   `.map()` over your dataset, except it provides convenience functions to retrieve
    any information you cache.
    
Robustness Gym ships with a few cached operations that you can use out-of-the-box. 

```python
from robustnessgym import Spacy, Stanza, TextBlob

# Create the Spacy CachedOperation
spacy_op = Spacy()

# Apply it on the "text" column of a dataset
dataset = spacy_op(batch_or_dataset=dataset, columns=["text"])

# Easily retrieve whatever information you need, wherever you need it

# Retrieve the tokens extracted by Spacy for the first 2 examples in the dataset
tokens = Spacy.retrieve(batch=dataset[:2], columns=["text"], proc_fns=Spacy.tokens)

# Retrieve everything Spacy cached for the first 2 examples, and process it yourself
spacy_info = Spacy.retrieve(batch=dataset[:2], columns=["text"])

# ...do stuff with spacy_info
```

### 3. Build slices
Robustness Gym supports a general set of
abstractions to create slices of data. Slices are just
datasets that are constructed by applying an instance of the `SliceBuilder` class
in Robustness Gym.
   
Robustness Gym currently supports slices of four kinds:
1. __Evaluation Sets__: slice constructed from a pre-existing dataset
2. __Subpopulations__: slice constructed by filtering a larger dataset
3. __Transformations__: slice constructed by transforming a dataset
4. __Attacks__: slice constructed by attacking a dataset adversarially

#### 3.1 Evaluation Sets
```python
from robustnessgym import Dataset, Slice

# Evaluation Sets: direct construction of a slice
boolq_slice = Slice(Dataset.load_dataset('boolq'))
```

#### 3.2 Subpopulations
```python
from robustnessgym import LengthSubpopulation
# A simple subpopulation that splits the dataset into 3 slices
# The intervals act as buckets: the first slice will bucket based on text with
# length between 0 and 4 
length_sp = LengthSubpopulation(intervals=[(0, 4), (8, 12), ("80%", "100%")])

# Apply it
dataset, slices, membership = length_sp(batch_or_dataset=dataset, columns=['text'])

# dataset is an updated dataset where every example is tagged with its slice
# slices are a list of Slice objects: think of this as a list of 3 datasets
# membership is a matrix of shape (n x 3) with 0/1 entries, assigning each of the n
# examples depending on whether they're in the slice or not 
```

#### 3.3 Transformations
```python
from robustnessgym import EasyDataAugmentation

# Easy Data Augmentation (https://github.com/jasonwei20/eda_nlp)
eda = EasyDataAugmentation(num_transformed=2)

# Apply it
dataset, eda_slices, eda_membership = eda(batch_or_dataset=dataset, columns=['text'])

# eda_slices is just 2 transformed versions of the original dataset
```
#### 3.4 Attacks
```python
from robustnessgym import TextAttack
from textattack.models.wrappers import HuggingFaceModelWrapper

# TextAttack
textattack = TextAttack.from_recipe(recipe='BAEGarg2019', 
                                    model=HuggingFaceModelWrapper(...))
```

### 4. Evaluate slices
At this point, you can just use your own code (e.g. in numpy) to calculate metrics
, since the slices are just datasets. 
  
```python
import numpy as np

def accuracy(true: np.array, pred: np.array):
    """
    Your function for computing accuracy.    
    """
    return np.mean(true == pred)

# Some model in your code
model = MyModel()

# Evaluation on the length slices
metrics = {}
for sl in slices:
    metrics[sl.identifier] = accuracy(true=sl["label"], pred=MyModel.predict(sl['text']))
```

Robustness Gym includes a `TestBench` abstraction to make this process easier.

```python
from robustnessgym import TestBench, Identifier, BinarySentiment

# Construct a testbench
testbench = TestBench(
    # Your identifier for the testbench
    identifier=Identifier(_name="MyTestBench"),
    # The task this testbench should be used to evaluate 
    task=BinarySentiment(),
)

# Add slices
testbench.add_slices(slices)

# Evaluate: Robustness Gym knows what metrics to use from the task
metrics = testbench.evaluate(model)
```

You can also get a Robustness Report using the TestBench.

```python
# Create the report
report = testbench.create_report(model)

# Generate the figures
_, figure = report.figures()
figure.write_image('my_figure.pdf', engine="kaleido")
```