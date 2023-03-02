
## 📉 ML with images in `meerkat`.

Let's do some machine learning on our Imagenette `DataFrame`.
We'll take a resnet18 pretrained on the full ImageNet dataset, perform inference on the validation set, and analyze the model's predictions and activations. 

The cell below downloads the model.. 

```{code-cell}
import torch
from torchvision.models import resnet18
import torchvision.transforms as transforms
model = resnet18(pretrained=True)
```

### 💈  Applying a transform to the images.
In order to do inference, we'll need to create a _new_ {class}`~meerkat.DeferredColumn`. The `ImageColumn` we defined above (_i.e._ `"img_path"`), does not apply any transforms after loading and simply returns a PIL image. Before passing the images through the model, we need to convert the PIL image to a `torch.Tensor` and normalize the color channels (along with a few other transformations). 

Note: the transforms defined below are the same as the ones used by torchvision, see [here](https://github.com/pytorch/examples/blob/cbb760d5e50a03df667cdc32a61f75ac28e11cbf/imagenet/main.py#L225). 

In the cell below, we specify a transform when creating the `ImageColumn`.

```{code-cell}
# Define transform
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]),
])

# Create new column with transform 
valid_df["input"] = valid_df["img"].defer(transform)
```

Notice that indexing this new column returns a `torch.Tensor`, not a PIL image...  
```{code-cell}
img = valid_df["input"][0]()
print(f"Indexing the `ImageColumn` returns an object of type: {type(img)}.")
```

... and that indexing a slice of this new column returns a {class}`~meerkat.TensorColumn`.
```{code-cell}
col = img = valid_df["input"][:3]()
print(f"Indexing a slice of the `ImageColumn` returns an object of type: {type(img)}.")
col
```

Let's see what the full `DataFrame` looks like now.  
```{code-cell}
valid_df.head()
```

### 💫 Computing model predictions and activations.
We'd like to perform inference and extract:
  
1. Output predictions  
2. Output class probabilities  
3. Model activations 

Note: in order to extract model activations, we'll need to use a [PyTorch forward hook](https://pytorch.org/tutorials/beginner/former_torchies/nnft_tutorial.html#forward-and-backward-function-hooks) and register it on the final layer of the ResNet. Forward hooks are just functions that get executed on the forward pass of a `torch.nn.Module`. 

```{code-cell}
# Define forward hook in ActivationExtractor class
class ActivationExtractor:
    """Extracting activations a targetted intermediate layer"""

    def __init__(self):
        self.activation = None

    def forward_hook(self, module, input, output):
        self.activation = output

# Register forward hook
extractor = ActivationExtractor()
model.layer4.register_forward_hook(extractor.forward_hook);
```

We want to apply a forward pass to each image in the `DataFrame` and store the outputs as new columns: `DataFrame.map` is perfectly suited for this task. 

```{code-cell}
# 1. Move the model to GPU, if available
# device = 0
device = "cpu"
model.to(device).eval()

# 2. Define a function that runs a forward pass over a batch 
@torch.no_grad()
def predict(input: mk.TensorColumn):
    x: torch.Tensor = input.data.to(device)  # We get the underlying torch tensor with `data` and move to GPU 
    out: torch.Tensor = model(x)  # Run forward pass

    # Return a dictionary with one key for each of the new columns. Each value in the
    # dictionary should have the same length as the batch. 
    return {
        "pred": out.cpu().numpy().argmax(axis=-1),
        "probs": torch.softmax(out, axis=-1).cpu(),
        "activation": extractor.activation.mean(dim=[-1,-2]).cpu()
    }
# 3. Apply the update. Note that the `predict` function operates on batches, so we set 
# `batched=True`. Also, the `predict` function only accesses the "input" column, by 
# specifying that here we instruct update to only load that one column and skip others 
pred_df = valid_df.map(function=predict, is_batched_fn=True, batch_size=32)
valid_df = mk.concat([valid_df, pred_df], axis="columns")
```

The predictions, output probabilities, and activations are now stored alongside the examples in the `DataFrame`. 

```{code-cell}
valid_df[["label_idx", "input", "pred", "probs", "activation"]].head()
```

### 🎯  Computing metrics and analyzing performance. 

Computing statistics on Meerkat `DataFrames` is straightforward because standard NumPy operators and functions can be applied directly to a `NumpyArrayColumn`. We take advantage of this below to compute the accuracy of the model.

```{code-cell}
valid_df["correct"] = valid_df["pred"] == valid_df["label_idx"].data
accuracy = valid_df["correct"].mean()
print(f"Micro accuracy across the ten Imagenette classes: {accuracy:0.3}")
```

Furthermore, since the `DataFrame` is naturally converted to a Pandas DataFrame, it's easy to use data visualization tools that interface with Pandas (_e.g._ seaborn, bokeh).

```{code-cell}
## OPTIONAL: this cell requires the seaborn dependency: https://seaborn.pydata.org/installing.html 
import seaborn as sns
import matplotlib.pyplot as plt
sns.barplot(data=valid_df.to_pandas(), y="label", x="correct");
```

### 🔎  Exploring model activations.
To better understand the behavior of our model, we'll explore the activations of the final convolutional layer of the ResNet. Recall that when we performed our forward pass, we extracted these activations and stored them in a new column called `"activation"`.

Unlike the the `NumpyArrayColumn`s we've been working with so far, the activation column has an additional dimension of size 512.

To visualize the activations, we'll use a dimensionality reduction technique ([UMAP](https://umap-learn.readthedocs.io/en/latest/)) to embed the activations in two dimensions. We'll store these embeddings in two new columns "umap_0" and "umap_1".

```{code-cell}
## OPTIONAL: this cell requires the umap dependency: https://umap-learn.readthedocs.io/en/latest/
!pip install umap-learn
from umap import UMAP

# 1. Compute UMAP embedding
reducer = UMAP()
embs = reducer.fit_transform(valid_df["activation"])

# 2. Add the embedding to the DataFrame as two new columns 
valid_df["umap_0"] = embs[:, 0]
valid_df["umap_1"] = embs[:, 1]

## OPTIONAL: this cell requires the seaborn dependency: https://seaborn.pydata.org/installing.html 
import seaborn as sns
import matplotlib.pyplot as plt
sns.scatterplot(data=valid_df.to_pandas(), x="umap_0", y="umap_1", hue="label");
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
```