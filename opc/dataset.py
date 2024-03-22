import ast
import json
import shutil, os
import os.path as osp

import pandas as pd
import numpy as np
import torch

from opc.utils.split import scaffold_split, similarity_split
from opc.utils.features import task_properties
from opc.utils.url import download_url


class PolymerDataset(object):
    def __init__(self, name="prediction", root="raw_data", transform="SMILES"):
        """
        Library-agnostic dataset object
            - root (str): the dataset folder will be located at root/prediction or root/generation
            - transform (str): SMILES or Fingerprint (default: SMILES) determines the type of data
        """

        self.name = name
        self.folder = osp.join(root, name)
        assert transform in ["SMILES", "Fingerprint"], "Invalid transform type"
        self.transform = transform

        self.task_type = name
        self.task_properties = task_properties[name]
        if self.task_properties is None:
            self.num_tasks = None
            self.eval_metric = "jaccard"
        else:
            self.num_tasks = len(self.task_properties)
            self.eval_metric = "wmae"

        self.url = f"https://github.com/open-polymer-challenge/data-dev/raw/main/{name}/data_dev.csv.gz"

        super(PolymerDataset, self).__init__()
        if transform == "SMILES":
            self.prepare_smiles()
        elif transform == "Fingerprint":
            self.prepare_fingerprints()

    def download(self):
        raw_file = osp.join(self.folder, "raw")
        download_url(self.url, raw_file)

    def get_idx_split(self, split_type="scaffold", to_list=False):
        path = osp.join(self.folder, "split", split_type)
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            split_dict = torch.load(os.path.join(path, "split_dict.pt"))
        else:
            if split_type == "scaffold":
                data_df = pd.read_csv(osp.join(self.folder, "raw", "data_dev.csv.gz"))
                train_idx, valid_idx, test_idx = scaffold_split(data_df, train_ratio=0.8, valid_ratio=None)
                train_idx = torch.tensor(train_idx, dtype=torch.long)
                valid_idx = torch.tensor(valid_idx, dtype=torch.long)
                test_idx = torch.tensor(test_idx, dtype=torch.long)
            elif split_type == "similarity":
                data_df = pd.read_csv(osp.join(self.folder, "raw", "data_dev.csv.gz"))
                if not os.path.exists('test_dev.json'):
                    raise FileNotFoundError(f"Similarity based splitting requires test_dev.json in {os.getcwd()}")
                with open('test_dev.json', 'r') as file:
                    test_dev = json.load(file)
                    test_dev = pd.json_normalize(test_dev)
                train_idx, valid_idx, test_idx = similarity_split(data_df, test_dev, train_ratio=0.9)
                train_idx = torch.tensor(train_idx, dtype=torch.long)
                valid_idx = torch.tensor(valid_idx, dtype=torch.long)
                test_idx = torch.tensor(test_idx, dtype=torch.long)
            else:
                raise ValueError("Invalid split type")

            os.makedirs(path, exist_ok=True)
            torch.save(
                {"train": train_idx, "valid": valid_idx, "test": test_idx},
                os.path.join(path, "split_dict.pt"),
            )
            split_dict = {"train": train_idx, "valid": valid_idx, "test": test_idx}

        if to_list:
            split_dict = {k: v.tolist() for k, v in split_dict.items()}
        return split_dict

    def get_task_weight(self, ids):
        if self.task_properties is not None:
            try:
                if not isinstance(self.labels, torch.Tensor):
                    labels = torch.stack(self.labels, dim=0)
                else:
                    labels = self.labels
                labels = labels[torch.LongTensor(ids)]
                task_weight = []
                for i in range(labels.shape[1]):
                    valid_num = labels[:, i].eq(labels[:, i]).sum()
                    task_weight.append(valid_num)
                task_weight = torch.sqrt(
                    1 / torch.tensor(task_weight, dtype=torch.float32)
                )
                return task_weight / task_weight.sum() * len(task_weight)
            except Exception as e:
                print(f"An error occurred: {e}")
        else:
            return None

    def prepare_smiles(self):
        raw_dir = osp.join(self.folder, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        if not osp.exists(osp.join(raw_dir, "data_dev.csv.gz")):
            self.download()
        data_df = pd.read_csv(osp.join(raw_dir, "data_dev.csv.gz"))

        processed_dir = osp.join(self.folder, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        x_list = []
        y_list = []
        for idx, row in data_df.iterrows():
            smiles = row["SMILES"]
            x_list.append(smiles)
            if self.task_properties is not None:
                y = []
                for task in self.task_properties:
                    y.append(float(row[task]))
                y = torch.tensor(y, dtype=torch.float32)
                y_list.append(y)
            else:
                y = torch.tensor(
                    ast.literal_eval(row["Conditions"]), dtype=torch.float32
                )
                y_list.append(y)

        self.data = x_list
        self.labels = y_list

    def prepare_fingerprints(self):
        raw_dir = osp.join(self.folder, "raw")
        os.makedirs(raw_dir, exist_ok=True)
        if not osp.exists(osp.join(raw_dir, "data_dev.csv.gz")):
            self.download()
        data_df = pd.read_csv(osp.join(raw_dir, "data_dev.csv.gz"))

        processed_dir = osp.join(self.folder, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        if not osp.exists(osp.join(processed_dir, "processed_fp.pt")):
            print("Processing fingerprints...")
            from rdkit import Chem
            from rdkit.Chem import AllChem

            x_list = []
            y_list = []
            for idx, row in data_df.iterrows():
                smiles = row["SMILES"]
                mol = Chem.MolFromSmiles(smiles)
                x = torch.tensor(
                    list(AllChem.GetMorganFingerprintAsBitVect(mol, 2)),
                    dtype=torch.int8,
                )
                x_list.append(x)
                if self.task_properties is not None:
                    y = []
                    for task in self.task_properties:
                        y.append(float(row[task]))
                    y = torch.tensor(y, dtype=torch.float32)
                    y_list.append(y)
                else:
                    y = torch.tensor(
                        ast.literal_eval(row["labels"]), dtype=torch.float32
                    )
                    y_list.append(y)
            x_list = torch.stack(x_list, dim=0)
            y_list = torch.stack(y_list, dim=0)
            torch.save((x_list, y_list), osp.join(processed_dir, "processed_fp.pt"))
        else:
            x_list, y_list = torch.load(osp.join(processed_dir, "processed_fp.pt"))

        self.data = x_list
        self.labels = y_list

    def __getitem__(self, idx):
        """Get datapoint(s) with index(indices)"""

        if isinstance(idx, (int, np.integer)):
            return self.data[idx], self.labels[idx]
        elif isinstance(idx, (list, np.ndarray)):
            return [self.data[i] for i in idx], [self.labels[i] for i in idx]
        elif isinstance(idx, torch.LongTensor):
            return self.data[idx], self.labels[idx]

        raise IndexError("Not supported index {}.".format(type(idx).__name__))

    def __len__(self):
        """Length of the dataset
        Returns
        -------
        int
            Length of Dataset
        """
        return len(self.data)

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, len(self))


if __name__ == "__main__":
    pass
