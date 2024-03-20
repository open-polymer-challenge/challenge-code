import numpy as np

try:
    import torch
except ImportError:
    torch = None

from rdkit import Chem
from rdkit.Chem import AllChem

from opc.utils.transform import scaling_error
from opc.utils.features import task_properties

import random

dim_reorder = list(range(256))
random.seed(0)
random.shuffle(dim_reorder)


def feature_extraction(mol):
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    if mol:
        fp1 = AllChem.GetMACCSKeysFingerprint(mol)
        fp1 = [int(b) for b in fp1.ToBitString()]
        dim2 = 256 - len(fp1)
        fp2 = list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=dim2))
        # concat fp1 and fp2
        fp = fp1 + fp2
        reordered_fp = [fp[dim] for dim in dim_reorder]
        return reordered_fp
    else:
        return None


class Evaluator:
    def __init__(self, name):
        assert name in ["prediction", "generation"], "Undefined evaluator name %s" % (
            name
        )
        self.name = name
        self.task_properties = task_properties[name]

        if "prediction" in self.name:
            self.num_tasks = len(self.task_properties)
            self.eval_metric = "wmae"
        else:
            self.num_tasks = 256
            self.eval_metric = "jaccard"

    def _parse_and_check_input(self, input_dict):
        if self.eval_metric == "wmae":
            if not "true" in input_dict:
                raise RuntimeError("Missing key of true")
            if not "pred" in input_dict:
                raise RuntimeError("Missing key of pred")

            true, pred = input_dict["true"], input_dict["pred"]

            """
                true: numpy ndarray or torch tensor of shape (num_data, num_tasks)
                pred: numpy ndarray or torch tensor of shape (num_data, num_tasks)
            """

            # converting to torch.Tensor to numpy on cpu
            if torch is not None and isinstance(true, torch.Tensor):
                true = true.detach().cpu().numpy()

            if torch is not None and isinstance(pred, torch.Tensor):
                pred = pred.detach().cpu().numpy()

            ## check type
            if not isinstance(true, np.ndarray):
                raise RuntimeError(
                    "Arguments to Evaluator need to be either numpy ndarray or torch tensor"
                )

            if not true.shape == pred.shape:
                raise RuntimeError("Shape of true and pred must be the same")

            if not true.ndim == 2:
                raise RuntimeError(
                    "true and pred mush to 2-dim arrray, {}-dim array given".format(
                        true.ndim
                    )
                )

            if not true.shape[1] == self.num_tasks:
                raise RuntimeError(
                    "Number of tasks for {} should be {} but {} given".format(
                        self.name, self.num_tasks, true.shape[1]
                    )
                )

            return true, pred

        elif self.eval_metric == "jaccard":

            if not "true" in input_dict:
                raise RuntimeError("Missing key of true")
            if not "generated" in input_dict:
                raise RuntimeError("Missing key of generated")

            true, generated = input_dict["true"], input_dict["generated"]

            if not isinstance(true, list):
                raise RuntimeError("true must be of type list")

            if not isinstance(generated, list):
                raise RuntimeError("generated must be of type list")

            if len(true) != len(generated):
                raise RuntimeError("Length of true and generated should be the same")

            true_mol, generate_mol = [], []
            for element in true:
                try:
                    true_mol.append(Chem.MolFromSmiles(element))
                except:
                    raise RuntimeError(f"SMILES string {element} in true is invalid")
            for element in generated:
                try:
                    generate_mol.append(Chem.MolFromSmiles(element))
                except:
                    generate_mol.append(None)
                    # raise RuntimeError(
                    #     f"SMILES string {element} in generated is invalid"
                    # )
            return true_mol, generate_mol

        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

    def validate(self, input_dict):

        if self.eval_metric == "wmae":
            task_weight = input_dict["task_weight"]
            input_dict = {"true": input_dict["true"], "pred": input_dict["pred"]}
            true, pred = self._parse_and_check_input(input_dict)
            return self._eval_wmae(true, pred, task_weight)
        elif self.eval_metric == "jaccard":
            true, pred = self._parse_and_check_input(input_dict)
            return self._eval_jaccard(true, pred)
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

    @property
    def expected_input_format(self):
        desc = "==== Expected input format of Evaluator for {}\n".format(self.name)
        if self.eval_metric == "wmae":
            desc += "{'true': true, 'pred': pred}\n"
            desc += (
                "- true: numpy ndarray or torch tensor of shape (num_data, num_tasks)\n"
            )
            desc += (
                "- pred: numpy ndarray or torch tensor of shape (num_data, num_tasks)\n"
            )
            desc += "where num_tasks is {}, and ".format(self.num_tasks)
            desc += "each row corresponds to one data.\n"
            desc += "nan values in true are ignored during evaluation.\n"
        elif self.eval_metric == "jaccard":
            desc += "{'true': true, 'generate': generate}\n"
            desc += "- true: a list of SMILES strings \n"
            desc += "- generate: a list of SMILES strings \n"
            desc += "where generated (valid) SMILES is from a generative model,\n"
            desc += "SMILES are then converted to Morgan Fingerprint \n"
            desc += "and Jaccard (Tanimoto) simiarlity is calculated for evaluation \n"
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

        return desc

    @property
    def expected_output_format(self):
        desc = "==== Expected output format of Evaluator for {}\n".format(self.name)
        if self.eval_metric == "wmae":
            desc += "{'wmae': wmae}\n"
            desc += "- wmae (float): weighted mean absolute error averaged across {} task(s)\n".format(
                self.num_tasks
            )
        elif self.eval_metric == "jaccard":
            desc += "{'jaccard': jaccard}\n"
            desc += "- jaccard (float): jaccard similarity score averaged over SMILES samples.\n"
        else:
            raise ValueError("Undefined eval metric %s " % (self.eval_metric))

        return desc

    def _eval_wmae(self, true, pred, task_weight):
        mae_list = []

        for i in range(true.shape[1]):
            is_labeled = true[:, i] == true[:, i]
            weight = float(task_weight[i])
            if is_labeled.sum() == 0:
                continue
            else:
                mae_list.append(
                    np.mean(
                        scaling_error(
                            np.abs(true[is_labeled, i] - pred[is_labeled, i]), i
                        )
                    )
                    * weight
                )
        if len(mae_list) == 0:
            raise RuntimeError("No labels")

        return {"wmae": sum(mae_list) / len(task_weight)}

    def _eval_jaccard(self, true_mol, generate_mol):
        fps_true = [
            feature_extraction(true_mol[idx])
            for idx, m in enumerate(generate_mol)
            if m is not None
        ]
        fps_generate = [feature_extraction(m) for m in generate_mol if m is not None]

        invalid_num = len(generate_mol) - len(fps_generate)

        intersection = [
            np.sum(np.array(fps_true[i]) & np.array(fps_generate[i]))
            for i in range(len(fps_true))
        ] + [0] * invalid_num
        union = [
            np.sum(np.array(fps_true[i]) | np.array(fps_generate[i]))
            for i in range(len(fps_true))
        ] + [1] * invalid_num

        jarccard = np.mean(np.array(intersection) / np.array(union))
        return {"jaccard": jarccard}


if __name__ == "__main__":
    pass
