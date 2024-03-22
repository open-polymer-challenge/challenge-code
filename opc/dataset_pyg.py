import os
import ast
import json
import os.path as osp
import pandas as pd

import torch
from torch_geometric.data import InMemoryDataset, Data

from opc.utils.mol import smiles2graph
from opc.utils.split import scaffold_split, similarity_split
from opc.utils.features import task_properties
from opc.utils.url import download_url


class PygPolymerDataset(InMemoryDataset):
    def __init__(
        self, name="prediction", root="raw_data", transform=None, pre_transform=None
    ):
        """
        - name (str): name of the dataset == prediction or generation
        - root (str): root directory to store the dataset folder
        - transform, pre_transform (optional): transform/pre-transform graph objects
        """

        self.name = name
        self.root = osp.join(root, name)

        self.task_type = name
        self.task_properties = task_properties[name]
        if self.task_properties is None:
            self.num_tasks = None
            self.eval_metric = "jaccard"
        else:
            self.num_tasks = len(self.task_properties)
            self.eval_metric = "wmae"

        self.url = f"https://github.com/open-polymer-challenge/data-dev/raw/main/{name}/data_dev.csv.gz"

        super(PygPolymerDataset, self).__init__(self.root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    def get_idx_split(self, split_type="scaffold"):
        path = osp.join(self.root, "split", split_type)
        if os.path.isfile(os.path.join(path, "split_dict.pt")):
            return torch.load(os.path.join(path, "split_dict.pt"))
        else:
            if split_type == "scaffold":
                data_df = pd.read_csv(osp.join(self.raw_dir, "data_dev.csv.gz"))
                train_idx, valid_idx, test_idx = scaffold_split(data_df, train_ratio=0.8, valid_ratio=None)
                train_idx = torch.tensor(train_idx, dtype=torch.long)
                valid_idx = torch.tensor(valid_idx, dtype=torch.long)
                test_idx = torch.tensor(test_idx, dtype=torch.long)
            elif split_type == "similarity":
                data_df = pd.read_csv(osp.join(self.raw_dir, "data_dev.csv.gz"))
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
        return {"train": train_idx, "valid": valid_idx, "test": test_idx}

    def get_task_weight(self, ids):
        if self.task_properties is not None:
            try:
                labels = self._data.y[torch.LongTensor(ids)]
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

    @property
    def raw_file_names(self):
        return ["data_dev.csv.gz"]

    @property
    def processed_file_names(self):
        return ["data_dev_processed.pt"]

    def download(self):
        download_url(self.url, self.raw_dir)

    def process(self):
        data_df = pd.read_csv(osp.join(self.raw_dir, "data_dev.csv.gz"))

        pyg_graph_list = []
        for idx, row in data_df.iterrows():
            smiles = row["SMILES"]
            graph = smiles2graph(smiles, add_fp=self.name == "prediction")

            g = Data()
            g.num_nodes = graph["num_nodes"]
            g.edge_index = torch.from_numpy(graph["edge_index"])

            del graph["num_nodes"]
            del graph["edge_index"]

            if graph["edge_feat"] is not None:
                g.edge_attr = torch.from_numpy(graph["edge_feat"])
                del graph["edge_feat"]

            if graph["node_feat"] is not None:
                g.x = torch.from_numpy(graph["node_feat"])
                del graph["node_feat"]

            try:
                g.fp = torch.tensor(graph["fp"], dtype=torch.int8).view(1, -1)
                del graph["fp"]
            except:
                pass

            if self.task_properties is not None:
                y = []
                for task in self.task_properties:
                    y.append(float(row[task]))
                g.y = torch.tensor(y, dtype=torch.float32).view(1, -1)
            else:
                g.y = torch.tensor(
                    ast.literal_eval(row["labels"]), dtype=torch.float32
                ).view(1, -1)

            pyg_graph_list.append(g)

        pyg_graph_list = (
            pyg_graph_list
            if self.pre_transform is None
            else self.pre_transform(pyg_graph_list)
        )
        print("Saving...")
        torch.save(self.collate(pyg_graph_list), self.processed_paths[0])

    def __repr__(self):
        return "{}()".format(self.__class__.__name__)


if __name__ == "__main__":
    pass
