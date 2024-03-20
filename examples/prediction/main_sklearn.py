import argparse
import numpy as np
from sklearn.impute import SimpleImputer

from opc import Evaluator, PolymerDataset
from dataset import TestDevPolymer

def validate(model, dataset, evaluator, task_weight):
    x, y = dataset
    x, y = x.numpy(), y.numpy()
    y_pred = model.predict(x)
    input_dict = {"true": y, "pred": y_pred, "task_weight": task_weight}
    return evaluator.validate(input_dict)


def main(seed):
    # Training settings
    parser = argparse.ArgumentParser(
        description="Machine learning baselines for polymer property prediction"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--method",
        type=str,
        default="RF",
        choices=["RF", "GP"],
    )
    parser.add_argument(
        "--no_print", action="store_true", help="no print if activated (default: False)"
    )

    args = parser.parse_args()

    ### automatic dataloading and splitting

    dataset = PolymerDataset(name="prediction", root="data_fp", transform="Fingerprint")
    split_idx = dataset.get_idx_split()
    train_weight = dataset.get_task_weight(split_idx["train"])
    valid_weight = dataset.get_task_weight(split_idx["valid"])

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator("prediction")

    train_x, train_y = dataset[split_idx["train"]]
    train_x, train_y = train_x.numpy(), train_y.numpy()

    if args.method == "RF":
        from sklearn.ensemble import RandomForestRegressor

        model = RandomForestRegressor(n_estimators=100, random_state=seed)
    elif args.method == "GP":
        from sklearn.gaussian_process import GaussianProcessRegressor
        from sklearn.gaussian_process.kernels import RBF, WhiteKernel

        model = GaussianProcessRegressor(
            kernel=RBF() + WhiteKernel(), random_state=seed
        )
    else:
        raise ValueError("Invalid method")

    ### training
    imputer = SimpleImputer(strategy="mean")
    train_y_imputed = imputer.fit_transform(train_y)
    model.fit(train_x, train_y_imputed)

    train_perf = validate(model, dataset[split_idx["train"]], evaluator, train_weight)
    valid_perf = validate(model, dataset[split_idx["valid"]], evaluator, valid_weight)

    best_epoch = -1
    best_train = train_perf[dataset.eval_metric]
    best_valid = valid_perf[dataset.eval_metric]

    print(
        "Finished. \n Best validation epoch {} with metric {}, train {:.4f}, valid {:.4f}".format(
            best_epoch, dataset.eval_metric, best_train, best_valid
        )
    )

    if seed == 0:
        test_dev = TestDevPolymer(name="prediction")
        test_feature, smiles_list, target_list = test_dev.prepare_feature(
            transform="Fingerprint"
        )
        save_prediction(model, test_feature, smiles_list, target_list, out_file=f"out-{args.method}.json")

    return (
        args.method,
        "prediction",
        dataset.eval_metric,
        best_train,
        best_valid,
        best_epoch,
    )

def save_prediction(model, test_feature, smiles_list, target_list, out_file="out.json"):
    from opc.utils.features import task_properties
    import json
    task_properties = task_properties['prediction']
    task_count = {}
    pred_json = []
    for idx, batch in enumerate(test_feature):
        y_pred = model.predict(np.array(batch).reshape(1, -1))
        targets = target_list[idx]
        entry = {"SMILES": smiles_list[idx]}
        for i, target in enumerate(targets):
            pred_value = y_pred[0,task_properties.index(target)]
            entry[target] = float(pred_value)
            task_count[target] = task_count.get(target, 0) + 1
        pred_json.append(entry)

    with open(out_file, "w") as f:
        json.dump(pred_json, f, indent=4)
    
    task_weight = np.sqrt(
        1 / np.array(list(task_count.values()))
    )
    task_weight = task_weight / task_weight.sum() * len(task_weight)

    print(
        f"Predictions saved to {out_file}, to be evaluated with weights {task_weight} for each task."
    )

if __name__ == "__main__":
    import os
    import pandas as pd

    results = {
        "model": [],
        "dataset": [],
        "seed": [],
        "metric": [],
        "train": [],
        "valid": [],
        "epoch": [],
    }
    df = pd.DataFrame(results)

    for i in range(10):
        model, dataset, metric, train, valid, epoch = main(i)
        new_results = {
            "model": model,
            "dataset": dataset,
            "seed": i,
            "metric": metric,
            "train": train,
            "valid": valid,
            "epoch": epoch,
        }
        df = pd.concat([df, pd.DataFrame([new_results])], ignore_index=True)

    if os.path.exists("result_each.csv"):
        df.to_csv("result_each.csv", mode="a", header=False, index=False)
    else:
        df.to_csv("result_each.csv", index=False)
    print(df)

    # Calculate mean and std, and format them as "mean±std".
    summary_cols = ["model", "dataset", "metric"]
    df_mean = df.groupby(summary_cols).mean().round(4)
    df_std = df.groupby(summary_cols).std().round(4)

    df_mean = df_mean.reset_index()
    df_std = df_std.reset_index()
    df_summary = df_mean[summary_cols].copy()
    # Format 'train', 'valid' columns as "mean±std".
    for col in ["train", "valid"]:
        df_summary[col] = df_mean[col].astype(str) + "±" + df_std[col].astype(str)

    # Save and print the summary DataFrame.
    if os.path.exists("result_summary.csv"):
        df_summary.to_csv("result_summary.csv", mode="a", header=False, index=False)
    else:
        df_summary.to_csv("result_summary.csv", index=False)
    print(df_summary)
