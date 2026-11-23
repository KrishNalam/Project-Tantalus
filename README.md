# Tantalus

Predict what a molecule tastes like (**sweet, bitter, umami, sour, salty, tasteless**) from its
structure with a graph neural network. The task is multi-label: one molecule can have several tastes.

## Approach

1. **Featurize.** SMILES strings are canonicalized with RDKit and turned into molecular graphs
   (atoms are nodes, bonds are edges in both directions) and 2048-bit Morgan fingerprints
   (radius 2).
2. **Data.** ChemTastesDB is the primary source, and BitterDB adds bitter labels. Molecules are
   deduplicated, conflicting labels are resolved, and every dropped row is logged.
3. **Split.** A Bemis–Murcko **scaffold split** keeps molecules that share a core structure on
   the same side of the split, so test molecules are structurally new to the model.
4. **Baselines.** Morgan fingerprints feed Random Forest and XGBoost. Any GNN has to beat these.
5. **GNNs.** GCN, then GIN, then an MPNN that uses bond features, built with PyTorch Geometric.
   Each label gets its own sigmoid output, trained with class-weighted binary cross-entropy.
6. **Explainability.** Atom-level attributions are drawn as heatmaps on the molecule.
7. **Demo.** A FastAPI endpoint takes a SMILES string and returns predictions plus an
   attribution image.

Results are reported as per-label ROC-AUC and PR-AUC, with macro averages, as mean ± std over
several seeds. Accuracy is never the headline number.

## Status

Phase 0 (featurization) is mostly done, and Phase 1 (data) is done. Phase 2 (baselines) is next. Live progress is
tracked in [`docs/roadmap.md`](docs/roadmap.md).

| Phase | Scope | State |
|---|---|---|
| 0 | Setup, featurization, exploration notebook | mostly done |
| 1 | Data loading, cleaning, merging, scaffold split | done (3728 molecules, scaffold splits for 5 seeds, class balance reported) |
| 2 | Fingerprint baselines | not started |
| 3 | GNNs | not started |
| 4 | Explainability | not started |
| 5 | Demo and write-up | not started |

## Setup

The project needs Python 3.12 and [uv](https://docs.astral.sh/uv/). It runs on CPU only, and
PyTorch is installed from the CPU wheel index.

```bash
uv sync
uv run python scripts/download_data.py   # fetches pinned, checksum-verified datasets into data/raw/
uv run python scripts/build_data.py      # cleaned table + scaffold splits into data/processed/
uv run jupyter lab                       # notebooks/00_explore.ipynb
```

Development:

```bash
uv run ruff check . && uv run ruff format .
uv run pytest
```

## Layout

```
scripts/download_data.py   pinned dataset downloads (ChemTastesDB, BitterDB)
scripts/build_data.py      Phase 1 pipeline: merged table, scaffold splits, class balance
src/tantalus/featurize.py  SMILES -> canonical SMILES, PyG graphs, Morgan fingerprints
src/tantalus/data.py       loaders, scaffold split, class-balance and data-quality reports
src/tantalus/models/       baselines and GNNs
notebooks/                 exploration, numbered by phase
docs/                      roadmap, design decisions (ADRs), learning log
```

## Data and licenses

- [ChemTastesDB](https://zenodo.org/records/15051366) v2.1: CC-BY-4.0
- [BitterDB](https://bitterdb.agri.huji.ac.il/) 2019: CC-BY-NC-ND-4.0. Its raw or merged data
  must not be redistributed, so the `data/` folder is never committed.

## Known limitations

- The scaffold split can't place acyclic or plain-benzene molecules in the test set. These two
  scaffold groups make up about 35% of the data.
- Salty and sour have too few molecules to evaluate on the scaffold split. They are reported only
  on a random split, as a deliberate side comparison.
- SMILES are stereo-free, so enantiomers count as one molecule.
