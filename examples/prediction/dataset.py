import urllib.request
import torch
import json
import os

class TestDevPolymer(object):
    def __init__(self, name="prediction"):
        self.url = f"https://raw.githubusercontent.com/open-polymer-challenge/data-dev/main/{name}/test_dev.json"
            
        super(TestDevPolymer, self).__init__()

        file_name = "test_dev.json"
        # check if the file exists
        if os.path.exists(file_name):
            print(f"File {file_name} already exists")
        else:
            try:
                with urllib.request.urlopen(self.url) as response:
                    data = response.read()
                    json_data = json.loads(data.decode("utf-8"))

                    # Write the JSON data to a local file
                    with open(file_name, "w") as json_file:
                        json.dump(json_data, json_file, indent=4)

                print(f"File successfully downloaded and saved as {file_name}")
            except Exception as e:
                print(f"An error occurred: {e}")

    def prepare_condition(self):
        import ast
        import numpy as np

        try:
            json_data = json.load(open("test_dev.json"))
        except:
            print("Error: File not found")
            return None
        condition_list = []
        for entry in json_data:
            conds = ast.literal_eval(entry["Condition"])
            condition_list.append(np.array(conds).astype(int))
        return condition_list

    def prepare_feature(self, transform):
        assert transform in ["SMILES", "Fingerprint", "PyG"], "Invalid transform type"
        try:
            json_data = json.load(open("test_dev.json"))
        except:
            print("Error: File not found")
            return None

        smiles_list = []
        target_list = []
        for entry in json_data:
            smiles_list.append(entry["SMILES"])
            targets = [key for key in entry.keys() if key != 'SMILES']
            target_list.append(targets)

        feature_list = []
        if transform == "Fingerprint":
            from rdkit import Chem
            from rdkit.Chem import AllChem

            for smiles in smiles_list:
                mol = Chem.MolFromSmiles(smiles)
                x = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2))
                feature_list.append(x)

        elif transform == "PyG":
            from opc.utils.mol import smiles2graph
            from torch_geometric.data import Data

            for smiles in smiles_list:
                graph = smiles2graph(smiles, add_fp=True)
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

                feature_list.append(g)

        elif transform == "SMILES":
            feature_list = smiles_list

        return feature_list, smiles_list, target_list


if __name__ == "__main__":
    pass
