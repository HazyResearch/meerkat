Installation
============================

This page describes how to get Robustness Gym installed and ready to use. Head to the
 [tutorials]() to start using Robustness Gym after installation.
 
Installing the Robustness Gym package
------------------
The only things you need to install to get setup.
### Install with pip 

``` shell
pip install robustnessgym
```


Optional Installation
--------------------------
The steps below aren't necessary unless you need these features.

#### Progress bars in Jupyter 
Enable the following Jupyter extensions to display progress bars properly. 
```shell
jupyter nbextension enable --py widgetsnbextension
jupyter labextension install @jupyter-widgets/jupyterlab-manager
```


#### TextBlob setup
To use TextBlob, download and install the TextBlob corpora.
```
python -m textblob.download_corpora
```



#### Installing Spacy GPU
To install Spacy with GPU support, use the installation steps given below.
``` shell
pip install cupy
pip install spacy[cuda]
python -m spacy download en_core_web_sm
```

#### Installing neuralcoref
The standard version of `neuralcoref` does not use GPUs for prediction and a [pull
 request]((https://github.com/huggingface/neuralcoref/pull/149)) that is pending adds this 
functionality. 
Follow the steps below to use this.   
```
git clone https://github.com/dirkgr/neuralcoref.git@754d470d484f56c5715ef35c220c217f28079eef
cd neuralcoref
git checkout GpuFix
pip install -r requirements.txt
pip install -e .
```


