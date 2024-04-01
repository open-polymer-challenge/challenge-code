import argparse
import random
from time import time
from typing import List, Optional

import joblib
import numpy as np

from joblib import delayed
from rdkit import Chem
from rdkit.Chem.rdchem import Mol

from opc import Evaluator, PolymerDataset
from opc.evaluate import feature_extraction as extractor
from utils import crossover, mutate
from dataset import TestDevPolymer


def compute_condition_similarity(train_conds, target_conds):
    intersection = np.sum(train_conds & target_conds, axis=1)
    union = np.sum(train_conds | target_conds, axis=1)
    return intersection / union


def condition_scoring(input_mol, target):
    feature = extractor(input_mol)
    feature = np.array([feature]).astype(int)
    similarity = compute_condition_similarity(feature, target)
    return similarity[0]


def make_mating_pool(population_mol: List[Mol], population_scores, offspring_size: int):
    """
    Given a population of RDKit Mol and their scores, sample a list of the same size
    with replacement using the population_scores as weights

    Args:
        population_mol: list of RDKit Mol
        population_scores: list of un-normalised scores given by ScoringFunction
        offspring_size: number of molecules to return

    Returns: a list of RDKit Mol (probably not unique)

    """
    # scores -> probs
    sum_scores = sum(population_scores)
    population_probs = [p / sum_scores for p in population_scores]
    mating_pool = np.random.choice(
        population_mol, p=population_probs, size=offspring_size, replace=True
    )
    return mating_pool


def reproduce(mating_pool, mutation_rate):
    """

    Args:
        mating_pool: list of RDKit Mol
        mutation_rate: rate of mutation

    Returns:

    """
    parent_a = random.choice(mating_pool)
    parent_b = random.choice(mating_pool)
    new_child = crossover(parent_a, parent_b)
    if new_child is not None:
        new_child = mutate(new_child, mutation_rate)
    return new_child


def sanitize(population_mol):
    new_population = []
    smile_set = set()
    for mol in population_mol:
        if mol is not None:
            try:
                smile = Chem.MolToSmiles(mol)
                if smile is not None and smile not in smile_set:
                    smile_set.add(smile)
                    new_population.append(mol)
            except ValueError:
                print("bad smiles")
    return new_population


class GraphGA_Generator(object):

    def __init__(
        self,
        train_smiles,
        train_conds,
        population_size,
        offspring_size,
        mutation_rate,
        n_jobs=-1,
        random_start=False,
        patience=5,
    ):
        self.pool = joblib.Parallel(n_jobs=n_jobs)
        self.train_smiles = train_smiles
        self.train_conds = train_conds
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.mutation_rate = mutation_rate
        self.random_start = random_start
        self.patience = patience

    def generate_optimized_molecules(
        self,
        target_conds,
        scoring_function,
        starting_population: Optional[List[str]] = None,
        iterations: Optional[int] = 5,
    ) -> List[str]:

        targe_len = target_conds.shape[0]
        final_generted_smiles = []
        for target_idx in range(targe_len):
            current_target = target_conds[target_idx]
            if self.random_start:
                starting_population = random.sample(
                    self.train_smiles, self.population_size
                )
            else:
                current_sim = compute_condition_similarity(
                    self.train_conds, current_target
                )
                sorted_idx = np.argsort(current_sim)[::-1]
                sorted_smiles = [self.train_smiles[i] for i in sorted_idx]
                starting_population = sorted_smiles[: self.population_size]

            population_mol = [Chem.MolFromSmiles(s) for s in starting_population]
            population_scores = self.pool(
                delayed(scoring_function)(m, current_target) for m in population_mol
            )
            t0 = time()
            patience = 0

            for iteration in range(iterations):
                # new_population
                mating_pool = make_mating_pool(
                    population_mol, population_scores, self.offspring_size
                )
                offspring_mol = self.pool(
                    delayed(reproduce)(mating_pool, self.mutation_rate)
                    for _ in range(self.population_size)
                )

                # add new_population
                population_mol += offspring_mol
                population_mol = sanitize(population_mol)

                # stats
                gen_time = time() - t0
                t0 = time()

                old_scores = population_scores
                population_scores = self.pool(
                    delayed(scoring_function)(m, current_target) for m in population_mol
                )
                population_tuples = list(zip(population_scores, population_mol))
                population_tuples = sorted(
                    population_tuples, key=lambda x: x[0], reverse=True
                )[: self.population_size]
                population_mol = [t[1] for t in population_tuples]
                population_scores = [t[0] for t in population_tuples]

                # early stopping
                if population_scores == old_scores:
                    patience += 1
                    print(f"Failed to progress: {patience}")
                    if patience >= self.patience:
                        print(f"No more patience, bailing...")
                        break
                else:
                    patience = 0

            generated_smiles = [Chem.MolToSmiles(m) for m in population_mol]
            generated_smiles = [
                s for s in generated_smiles if s not in starting_population
            ]
            if len(generated_smiles) == 0:
                print(f"No new molecules generated for {target_idx}")
                final_generted_smiles.append(None)
            else:
                final_generted_smiles.append(generated_smiles[0])

            print(
                f"Generation idx {target_idx} | "
                f"{iteration} | "
                f"max: {np.max(population_scores):.3f} | "
                f"avg: {np.mean(population_scores):.3f} | "
                f"min: {np.min(population_scores):.3f} | "
                f"std: {np.std(population_scores):.3f} | "
                f"{gen_time:.2f} sec/gen "
            )

        return final_generted_smiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--population_size", type=int, default=100)
    parser.add_argument("--offspring_size", type=int, default=200)
    parser.add_argument("--mutation_rate", type=float, default=0.01)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--random_start", action="store_true")
    parser.add_argument("--patience", type=int, default=2)
    args = parser.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)

    dataset = PolymerDataset(name="generation", root="data_smiles", transform="SMILES")
    split_idx = dataset.get_idx_split(to_list=True)
    train_smiles, train_conds = dataset[split_idx["train"]]
    valid_smiles, valid_conds = dataset[split_idx["valid"]]

    train_conds = [t.numpy() for t in train_conds]
    valid_conds = [t.numpy() for t in valid_conds]
    train_conds = np.vstack(train_conds).astype(int)
    valid_conds = np.vstack(valid_conds).astype(int)

    evaluator = Evaluator("generation")

    generator = GraphGA_Generator(
        train_smiles,
        train_conds,
        population_size=args.population_size,
        offspring_size=args.offspring_size,
        mutation_rate=args.mutation_rate,
        n_jobs=args.n_jobs,
        random_start=args.random_start,
        patience=args.patience,
    )

    valid_generations = generator.generate_optimized_molecules(
        valid_conds,
        scoring_function=condition_scoring,
        iterations=args.iterations,
    )
    input_dict = {"true": valid_smiles, "generated": valid_generations}
    performance = evaluator.validate(input_dict)
    print("Valid Performance:", performance[dataset.eval_metric])

    test_dev = TestDevPolymer(name="generation")
    test_conds = test_dev.prepare_condition()
    test_conds = np.vstack(test_conds).astype(int)
    test_generations = generator.generate_optimized_molecules(
        test_conds, scoring_function=condition_scoring, iterations=args.iterations
    )

    import json
    
    save_conditions = [str(condition.tolist()) for condition in test_conds]
    data_to_save = []
    for smiles, conditions in zip(test_generations, save_conditions):
        entry = {"Conditions": conditions, "SMILES": smiles}
        data_to_save.append(entry)

    filename = f"out-GraphGA-{args.random_start}.json"
    with open(filename, "w") as json_file:
        json.dump(data_to_save, json_file, indent=4)


if __name__ == "__main__":
    main()
