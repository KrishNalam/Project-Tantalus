"""Train a GNN on every saved split and save val/test predictions.

Usage:
    uv run python scripts/build_data.py              # once, writes the splits
    uv run python scripts/run_gnn.py gcn
    uv run python scripts/run_gnn.py gin
    uv run python scripts/run_gnn.py gine   # bond-aware MPNN
    uv run python scripts/run_gnn.py gin_mean   # GIN with mean pooling
    uv run python scripts/run_gnn.py gcn --split random
    uv run python scripts/run_gnn.py gin_mean --min-epochs 20  # -> gin_mean_min20

Same output format as ``scripts/run_baseline.py``, so ``evaluate.py`` scores
baselines and GNNs identically: ``data/processed/predictions/<model>.csv``
(scaffold split) or ``<model>_random.csv`` (random-split comparison).

The best-val weights of every seed's model are also saved to
``data/processed/checkpoints/<model>[_random]_seed<k>.pt`` (gitignored), so
Phase 4 can explain the exact models that produced the predictions; reload
them with ``models.gnn.load_checkpoint``.

``--min-epochs N`` (N > 1) turns on the minimum-epoch guard in
``train.train_gnn`` and appends ``_min<N>`` to every output name, so guarded
runs never overwrite the unguarded ones they are compared against.

Training runs on one CPU thread (``train.make_reproducible``), so rerunning
this script reproduces the same checkpoints bit for bit.
"""

import argparse

import pandas as pd
import torch

from tantalus.data import LABELS, PROCESSED_DIR, SPLIT_FILES
from tantalus.models.gnn import MODELS
from tantalus.train import (
    load_graph_table,
    make_reproducible,
    predict_proba,
    train_gnn,
)


def main() -> None:
    """Train one model per split seed (early stopping on val), predict val/test."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("model", choices=sorted(MODELS))
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="scaffold")
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=1,
        help="first epoch eligible for early stopping's best epoch (1 = off)",
    )
    args = parser.parse_args()

    splits_file, _ = SPLIT_FILES[args.split]
    suffix = "" if args.split == "scaffold" else f"_{args.split}"
    if args.min_epochs > 1:
        suffix += f"_min{args.min_epochs}"
    checkpoint_dir = PROCESSED_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
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
        make_reproducible(seed)  # weight init + pinned threads, see train.py
        model, best_epoch = train_gnn(
            MODELS[args.model](),
            part["train"],
            part["val"],
            seed=seed,
            min_epochs=args.min_epochs,
        )
        torch.save(
            {
                "model": args.model,
                "split": args.split,
                "seed": seed,
                "best_epoch": best_epoch,
                "min_epochs": args.min_epochs,
                "state_dict": model.state_dict(),
            },
            checkpoint_dir / f"{args.model}{suffix}_seed{seed}.pt",
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
    out_path = out_dir / f"{args.model}{suffix}.csv"
    pd.concat(frames, ignore_index=True).to_csv(out_path, index=False)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
