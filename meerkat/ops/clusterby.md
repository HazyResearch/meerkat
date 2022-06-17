
# ClusterBy
```
dp = mk.dataset.get(")
dp.clusterby(by="image")
```


registry of "clusterers"
"kmeans"
"gmm"
"hierarchical"
"spectral"
```
def cluster(
    data: Union[AbstractColumn, DataPanel], 
    method: Union[str, sklearn.base.ClusterMixin] ="kmeans", 
    encoder: str = "auto", 
    return_clusterer: bool = False,
    **kwargs
) -> NumpyArrayColumn, ClusterMixin:
    """
    """
    if isinstance(method, str):
        method = getattr(sklearn.cluster, method)(**kwargs)

    if return_clusterer:
        return method.fit_predict(data), method
    else:
        return method.fit_predict(data)
```
k = 3
[1, 1, 0, 2, 1, 0, 2]


```
def _cluster(data: Union[AbstractColumn, DataPanel], method: sklearn.base.ClusterMixin ="kmeans", **kwargs):
    """
    The magic happens here...
    """
```

```
def clusterby(data: DataPanel, by: , method: Union[str, sklearn.base.ClusterMixin]) -> ClusterBy:
    cluster(data[by])
```

```
(1)
cb = clusterby(dp, by=["image"])
cb["date"].mean() 


(2)
dp["clusters"] = cluster(dp["image"])

cb = clusterby(dp, by=["image"])
cb.to_dp()

```


```
dp["clusters"] = cluster_algorithm.fit_predict(dp, ["image"], algorithm = 'k_means')
dp.groupby("clusters")
```
```
dp.clusterby(by="image")
````

```python 
dp.groupby(by='label')
```




## How to deal with embeddings?
When we do a clusterby over unstructured data (e.g. images, audio), its not totally clear
feature representations to use. 

Ideally abstract feature reps away from user. 

cb = dp.clusterby(by="image", encoder="auto")

Open research question: how to select the right embedding for the by data? 

How to deal with caching and making sure that we don't repeat computation? 

```
cb = dp.clusterby("image")

# this second run, should not recompute the embeddings 
cb = dp.clusterby("image", method="gmm")

# lots of ops are going to depend on embeddings, so cache should be shared 
cb = dp.sliceby("image")
```
Approaches:
1. Create some cache variable for clusterby 
2. Store it in a new column (invisible, like dot files in os) – name the cache hash if you will 
    if you embed image column with clip, then a new column is created called "clip(image)"



Create a separate `embed` module. Presumably, clusterby is going `embed` under the hood. 


Potentially a danger in abstract away embedding, if the user thinks the embedding is more powerful than it actually is. 
- There is not just one best embedding – and there are many axes along which people will prefer an embedding
    - Cost
    - Speed
    - Potential for social bias 
    - Specific domains (e.g. medical etc.)
    - Modality (e.g. )
    - Interpretability (e.g. is it aligned with language)
    - Harder to specify things. (Language is the most intuitive medium for expressing these desires, but also could draw )
        - e.g. Cluster images by their backgrounds 
        - e.g. 

- What could the interface look like:
```
    cb = dp.clusterby(
        "image", 
        instruction="group images by their background"
    )

    cb = dp.clusterby(
        "image", 
        description="photo with a {} background", 
        instruction="group images by their background", crop = "filename.png", lt = (500, 400), rb = ( 600, 450)
    )

    clusterby(
        "image",
        instruction=""
    )

    

    # instruction should be like an abstrtaction independent embedding, should work with language model 
    # TODO: we 

    class Instruction: 
        def __init__(self):
            self.lang_instruction = None
           

    class ImageInstruction:
         self.im_instruction = 
            "Crop"
            self.rb = ()
            self.lt = ()

    
    # lt, rb could be generalized to a polygon but that's kinda extra.

    dp["image].label(description="sunny background")

    ```


    This instruction tells the clusterer how to pick an embedding algorithm.

    1. N images, derscription = "background"
    2. embed_col = images.map { $0.embed_by("background") }
        - this is really hard. 
        - CLIP: 
    3. ... 

    
    img_emb = [1, 2, 1, 4]  # this is one image
    text_emb = [1, 5, 0, 0]  # this is the embedding "background"
    
    img_emb has D dimensions. which dimensions 0..<D correspond to the background. i.e changes in those dimensions will mean you have a different background.

    idea for solving:
    computing gradient of dot product wrt image embeddings.

    which dimensions if i twiddle them change your background score. 
    
    Once we have the gradient, take the top 50 dimensions with the highest absolute value, and only use those for the embedding.
    
    def new_embed(x):
        return embed(x)[gradient_mask]