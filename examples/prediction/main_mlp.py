import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils import parameters_to_vector, vector_to_parameters

import argparse
from tqdm.auto import tqdm

from opc import Evaluator, PolymerDataset
from model import MLP
from dataset import TestDevPolymer

criterion = torch.nn.L1Loss()

def seed_torch(seed=0):
    print("Seed", seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def training(model, device, loader, optimizer):
    model.train()

    # for step, batch in enumerate(tqdm(loader, desc="Iteration")):
    for step, batch in enumerate(loader):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device).to(torch.float32)
        pred = model(x)
        is_valid = ~torch.isnan(y)
        loss = criterion(pred[is_valid], y[is_valid])
        loss.backward()
        optimizer.step()


def validate(model, device, loader, evaluator, task_weight):
    model.eval()
    y_true = []
    y_pred = []

    for step, batch in enumerate(loader):
        x, y = batch
        x = x.to(device).to(torch.float32)
        y = y.to(device).to(torch.float32)
        pred = model(x)
        y_true.append(y.detach().cpu())
        y_pred.append(pred.detach().cpu())

    y_true = torch.cat(y_true, dim=0).numpy()
    y_pred = torch.cat(y_pred, dim=0).numpy()

    input_dict = {"true": y_true, "pred": y_pred, "task_weight": task_weight}

    return evaluator.validate(input_dict)

def main(seed):
    # Training settings
    parser = argparse.ArgumentParser(
        description="MLP baseline for polymer property prediction"
    )
    parser.add_argument(
        "--device", type=int, default=0, help="which gpu to use if any (default: 0)"
    )
    parser.add_argument(
        "--emb_dim",
        type=int,
        default=300,
        help="dimensionality of hidden units (default: 300)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="input batch size for training (default: 32)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="number of epochs to train (default: 300)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=50,
        help="number of epochs to stop training(default: 50)",
    )
    parser.add_argument(
        "--num_workers", type=int, default=0, help="number of workers (default: 0)"
    )
    parser.add_argument(
        "--no_print", action="store_true", help="no print if activated (default: False)"
    )

    args = parser.parse_args()

    device = (
        torch.device("cuda:" + str(args.device))
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    ### automatic dataloading and splitting

    dataset = PolymerDataset(name="prediction", root = 'data_fp', transform="Fingerprint")
    split_idx = dataset.get_idx_split()
    train_weight = dataset.get_task_weight(split_idx["train"])
    valid_weight = dataset.get_task_weight(split_idx["valid"])

    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator("prediction")

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

    model = MLP(2048, hidden_features=4 * args.emb_dim, out_features=dataset.num_tasks).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_train, best_valid, best_params = None, None, None
    best_epoch = 0
    print("Start training...")
    for epoch in range(args.epochs):
        training(model, device, train_loader, optimizer)
        valid_perf = validate(model, device, valid_loader, evaluator, valid_weight)
        if epoch == 0 or valid_perf[dataset.eval_metric] < best_valid:
            train_perf = validate(model, device, train_loader, evaluator, train_weight)
            best_params = parameters_to_vector(model.parameters())
            best_valid = valid_perf[dataset.eval_metric]
            best_train = train_perf[dataset.eval_metric]
            best_epoch = epoch

            if not args.no_print:
                print(
                    "Update Epoch {}: best_train: {:.4f} best_valid: {:.4f}".format(
                        epoch, best_train, best_valid
                    )
                )
        else:
            if not args.no_print:
                print(
                    "Epoch {}: best_valid: {:.4f}, current_valid: {:.4f}, patience: {}/{}".format(
                        epoch,
                        best_valid,
                        valid_perf[dataset.eval_metric],
                        epoch - best_epoch,
                        args.patience,
                    )
                )
            if epoch - best_epoch > args.patience:
                break

    print(
        "Finished. \n Best validation epoch {} with metric {}, train {:.4f}, valid {:.4f}".format(
            best_epoch, dataset.eval_metric, best_train, best_valid
        )
    )

    vector_to_parameters(best_params, model.parameters())

    if seed == 0:
        test_dev = TestDevPolymer(name="prediction")
        test_feature, smiles_list, target_list = test_dev.prepare_feature(
            transform="Fingerprint"
        )
        save_prediction(model, device, test_feature, smiles_list, target_list, out_file=f"out-mlp.json")

    return (
        'mlp',
        "prediction",
        dataset.eval_metric,
        best_train,
        best_valid,
        best_epoch,
    )

def save_prediction(model, device, test_feature, smiles_list, target_list, out_file="out.json"):
    from opc.utils.features import task_properties
    import json
    task_properties = task_properties['prediction']
    task_count = {}
    pred_json = []
    test_feature = torch.tensor(test_feature, dtype=torch.float32)
    loader = DataLoader(test_feature, batch_size=1, shuffle=False)
    for idx, batch in enumerate(loader):
        batch = batch.to(device)
        y_pred = model(batch)
        targets = target_list[idx]
        entry = {"SMILES": smiles_list[idx]}
        for i, target in enumerate(targets):
            pred_value = y_pred.detach().cpu().numpy()[0, task_properties.index(target)]
            entry[target] = float(pred_value)
            task_count[target] = task_count.get(target, 0) + 1
        pred_json.append(entry)

    with open(out_file, "w") as f:
        json.dump(pred_json, f, indent=4)
    
    task_weight = torch.sqrt(
        1 / torch.tensor(list(task_count.values()), dtype=torch.float32)
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
        "test": [],
        "epoch": [],
    }
    df = pd.DataFrame(results)

    for i in range(10):
        seed_torch(i)
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
