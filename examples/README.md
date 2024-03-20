# Baseline and examples

1. Baseline code for polymer property prediction, including methods GNN, MLP, Random Forest, Gaussian Process, and pretrained representation from chemical language model and GNNs.

2. Baseline code for conditional polymer generation with graph genetic algorithm (GraphGA).

## Requirements

This code package was developed and tested with Python 3.11.7 and PyTorch 2.1.0+cu118. All dependencies specified in the ```requirements.txt``` file. The packages can be installed by
```
pip install -r requirements.txt
```

Or following the instructions to create environment for both prediction and generation


```
conda create --name opc_env python=3.11.7
conda activate opc_env

pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install torch_geometric
pip install torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html
pip install pyarrow pandas rdkit ogb PyYAML

```
For pretrained representation, the following package is necessary
```
pip install transformers molfeat dgllife dgl
```

## Usage for prediction

Please enter the corresponding folder by `cd prediction`.

Following is an example to run experiments:

```
python main_gnn.py
```

Here is one way to generate training log file

```
nohup python -u main_gnn.py > gnn.log
```

## Usage for generation

Please enter the corresponding folder by `cd generation`.

Following is an example to run experiments:
```
python graphga.py
```

Here is one way to generate training log file

```
nohup python -u graphga.py > graphga.log
```


## Output for test-dev

The dataset class defined in `dataset.py` provides a method for automatically downloading the `test-dev.json` file.

The function or code examples offers a way to fill the JSON file. Results are saved to the file `out-{method}.json`, which can be used for submission and evaluation. Please do not use the data in the `test-dev.json` set for model training at any phase.