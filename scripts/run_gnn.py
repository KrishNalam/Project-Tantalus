"""Train a GNN on every saved split and save val/test predictions.

Usage:
    uv run python scripts/build_data.py              # once, writes the splits
    uv run python scripts/run_gnn.py gcn
    uv run python scripts/run_gnn.py gin
    uv run python scripts/run_gnn.py gcn --split random

Same output format as ``scripts/run_baseline.py``, so ``evaluate.py`` scores
baselines and GNNs identically: ``data/processed/predictions/<model>.csv``
(scaffold split) or ``<model>_random.csv`` (random-split comparison).
"""

import argparse

import pandas as pd
import torch

from tantalus.data import LABELS, PROCESSED_DIR, SPLIT_FILES
from tantalus.models.gnn import GCN, GIN
from tantalus.train import load_graph_table, predict_proba, train_gnn

MODELS = {"gcn": GCN, "gin": GIN}


def main() -> None:
    """Train one model per split seed (early stopping on val), predict val/test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="scaffold")
    args = parser.parse_args()

    splits_file, _ = SPLIT_FILES[args.split]
    graphs, splits = load_graph_table(splits_file=splits_file)
    seed_columns = [c for c in splits.columns if c.startswith("seed_")]
    frames = []
    for column in seed_columns:
        seed = int(column.removeprefix("seed_"))
        split = splits[column].to_numpy()
        part = {
            name: [g for g, s in zip(graphs, split) if s == name]
            for name in ("train", "val")
        }
        torch.manual_seed(seed)  # weight initialisation
        model, best_epoch = train_gnn(
            MODELS[args.model](), part["train"], part["val"], seed=seed
        )

        evaluated = split != "train"
        eval_graphs = [g for g, keep in zip(graphs, evaluated) if keep]
        probs = predict_proba(model, eval_graphs)
        frame = pd.DataFrame(
            {
                "canonical_smiles": splits["canonical_smiles"][evaluated].to_numpy(),
                "seed": seed,
                "split": split[evaluated],
            }
        )
        y_true = torch.cat([g.y for g in eval_graphs]).int().numpy()
        for j, label in enumerate(LABELS):
            frame[f"{label}_true"] = y_true[:, j]
            frame[f"{label}_prob"] = probs[:, j]
        frames.append(frame)
        print(
            f"seed {seed}: trained on {len(part['train'])}, best epoch {best_epoch}, "
            f"predicted {evaluated.sum()}"
        )

    out_dir = PROCESSED_DIR / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "scaffold" else f"_{args.split}"
    out_path = out_dir / f"{args.model}{suffix}.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
