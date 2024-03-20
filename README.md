<p align='center'>
  <img width='60%' src='https://i.ibb.co/DpvbmQg/page.png' />
</p>

--------------------------------------------------------------------------------

## Overview

The Open Polymer Challenge (OPC) targets two application scenarios in polymer discovery: polymer property prediction and polymer generation, which are crucial for virtual screening and inverse design. To facilitate the challenge, the package aims to automatically download data, preprocess it, and evaluate models.

## Installation


You can install package using `pip install opc`. Current version of package is tested with the following requirements
 - joblib==1.3.2
 - numpy==1.26.4
 - pandas==2.2.1
 - tqdm==4.66.2
 - rdkit==2023.9.5
 - torch==2.1.0+cu118
 - torch_geometric==2.5.1

You can also install the package from source
```bash
git clone https://github.com/open-polymer-challenge/challenge-code
cd challenge-code
pip install -e .
```

## Package Usage

Please enter the folder `examples` to see more details on the usage of the package in specific models. The two major usages of the package include: (1) data downloading and preprocessing, and (2) standard evaluation for the challenge.

Example with Pytorch geometric is
```python
from opc import Evaluator, PygPolymerDataset
from torch_geometric.loader import DataLoader

# Create a dataset by transforming the polymer into PyG graph data, then obtain splitting and evaluation weights.
dataset = PygPolymerDataset(name="prediction", root="data_pyg")
split_idx = dataset.get_idx_split()
train_weight = dataset.get_task_weight(split_idx["train"])
valid_weight = dataset.get_task_weight(split_idx["valid"])

train_loader = DataLoader(
    dataset[split_idx["train"]],
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)
valid_loader = DataLoader(
    dataset[split_idx["valid"]],
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)
# Define evaluation for the prediction task
evaluator = Evaluator("prediction")

# Model training ...

# Model evaluation
input_dict = {"true": y_true, "pred": y_pred, "task_weight": task_weight}
result = evaluator.validate(input_dict)
```

Example with Pytorch only
```python
from opc import Evaluator, PolymerDataset
from torch.utils.data import DataLoader, TensorDataset

# Create a dataset by transforming the polymer into fingerprint vectors as input features, then obtain splitting and evaluation weights.
dataset = PolymerDataset(name="prediction", root = 'data_fp', transform="Fingerprint")
split_idx = dataset.get_idx_split()
train_weight = dataset.get_task_weight(split_idx["train"])
valid_weight = dataset.get_task_weight(split_idx["valid"])

train_loader = DataLoader(
    TensorDataset(*dataset[split_idx["train"]]),
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=args.num_workers,
)
valid_loader = DataLoader(
    TensorDataset(*dataset[split_idx["valid"]]),
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=args.num_workers,
)

# Define evaluation for the prediction task
evaluator = Evaluator("prediction")

# Model training ...

# Model evaluation
input_dict = {"true": y_true, "pred": y_pred, "task_weight": task_weight}
result = evaluator.validate(input_dict)

```

## Challenge Submission

The [test-dev data](https://github.com/open-polymer-challenge/data-dev/tree/main) could be found online. Here is an example for the entries in the `test-dev.json` in the prediction track:

```json
[
    {
        "SMILES": "*Nc1ccc([C@H](CCCC)c2ccc(C3(c4ccc([C@@H](CCCC)c5ccc(N*)cc5C)cc4)CCC(C)CC3)cc2)c(C)c1",
        "FFV": null
    },
    {
        "SMILES": "*Nc1ccc([C@H](CCCCCC)c2ccc(C3(c4ccc([C@@H](CCCCCC)c5ccc(N*)cc5)cc4)CCC(CCCCC)CC3)cc2)cc1",
        "FFV": null
    },
    {
        "SMILES": "*Nc1ccc(/C=C/c2ccc(N*)cc2-c2ccccc2)c(-c2ccccc2)c1",
        "FFV": null
    },
    {
        "SMILES": "*c1ccc(-c2ccc(Oc3ccc(C4(c5ccc(Oc6ccc(-c7ccc(N8C(=O)c9ccc(Oc%10ccc%11c(c%10)C(=O)N(*)C%11=O)cc9C8=O)cc7)cc6C(F)(F)F)cc5)c5ccccc5C(=O)N4c4ccccc4)cc3)c(C(F)(F)F)c2)cc1",
        "O2": null,
        "N2": null,
        "CO2": null,
        "CH4": null
    },
    // ...
]
```

Here is an example for the entries in the generation track:
```json
[
    {
        "Conditions": "[0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0]",
        "SMILES": null
    },
    {
        "Conditions": "[0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0]",
        "SMILES": null
    },
    // ...
]
```
Please fill the `null` values for submission. 

The `TestDevPolymer` function in the example code `dataset.py` is one way for automatic downloading and usage of `test-dev.json`:

```python

# Used for prediction with PyG, where the test_feature is a list of PyG objects and the target_list is the dimension index of the prediction task
test_dev = TestDevPolymer(name="prediction")
test_feature, smiles_list, target_list = test_dev.prepare_feature(
    transform="PyG"
)

# Used for generation 
test_dev = TestDevPolymer(name="generation")
test_conds = test_dev.prepare_condition()
```


## Acknowledgments
The package is developed based on [Open Graph Benchmark (OGB)](https://github.com/snap-stanford/ogb). We appreciate their open-source contribution to the community.
