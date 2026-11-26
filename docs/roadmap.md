# Roadmap

## Phase 0 — Setup & featurization Status: in progress

- [x] CPU PyTorch installs and imports (`import torch, torch_geometric`)
- [x] Project skeleton exists (`data/{raw,processed}`, `notebooks/`, `docs/`, `src/tantalus/models/`)
- [x] `featurize.py`: `canonicalize` / `atom_features` / `bond_features` / `smiles_to_graph` / `smiles_to_fingerprint`
- [x] `notebooks/00_explore.ipynb` runs end-to-end
- [ ] `tests/test_featurize.py` written (by me) and reviewed
- [x] Datasets downloaded to `data/raw/` (ChemTastesDB v2.1 and BitterDB 2019 via `scripts/download_data.py`; SweetenersDB dropped by owner decision 2026-11-22: hosts dead and its data is already in ChemTastesDB via Bouysset2020)

Done when: the featurizer works on a known molecule and the notebook demonstrates
canonicalization, atom features, fingerprints, and Tanimoto similarity.

---

## Phase 1 — Data Status: done

- [x] Load ChemTastesDB (primary), BitterDB, SweetenersDB (`data.load_chemtastesdb`, `data.load_bitterdb`; SweetenersDB dropped, see Phase 0)
- [x] Canonicalize, dedup across sources, resolve label conflicts, drop invalids (documented) (`assign_labels` → `strip_salts` → `merge_duplicates`; every drop in `dropped.csv`, every change in `salt_log.csv` / `merge_log.csv`)
- [x] Merge into one table: canonical SMILES + multi-hot labels (`data.build_dataset` → `data/processed/molecules.csv`, 3728 molecules)
- [x] Bemis–Murcko scaffold split implemented (`data.murcko_scaffold` + `data.scaffold_split`; run on the merged table for seeds 0–4 by `data.write_splits` → `data/processed/splits.csv`)
- [x] Class balance per split reported (`scripts/build_data.py` → `data/processed/split_balance.csv`; min positives over seeds 0–4 in val/test: sweet 103/95, bitter 147/184, umami 8/9, tasteless 10/6, sour 0/2, salty 0/0, see ADR 2026-11-23)

Done when: a single processed table + scaffold split exists with every drop documented
and class balance reported.

---

## Phase 2 — Baselines Status: in progress

- [x] Morgan fingerprints → Random Forest (`models/baselines.py` + `scripts/run_baseline.py random_forest`: one class-weighted forest per label, seeds 0–4, val/test probabilities → `data/processed/predictions/random_forest.csv`; scores pending the owner's `evaluate.py`)
- [x] Morgan fingerprints → XGBoost (`baselines.train_xgboost` + `scripts/run_baseline.py xgboost`: one `scale_pos_weight`-weighted model per label, default hyperparameters, seeds 0–4 → `data/processed/predictions/xgboost.csv`; CPU-only `xgboost-cpu` wheel)
- [x] Random split saved as a deliberate comparison, the only place salty/sour are scored (`data.random_split` → `random_splits.csv`; `run_baseline.py <model> --split random` → `predictions/<model>_random.csv`; see ADR 2026-11-23)
- [ ] Scaffold split, multiple seeds, per-label ROC-AUC / PR-AUC (mean ± std)

Done when: baseline metrics are recorded on the scaffold split as the number the GNN must beat.

---

## Phase 3 — GNN Status: in progress

- [x] GCN (`models/gnn.py` + `train.py` + `scripts/run_gnn.py gcn`: seeds 0–4, early stopping on val loss, train-only `pos_weight` → `data/processed/predictions/gcn.csv`, same format as the baselines; scores pending the owner's `evaluate.py`)
- [x] GIN (`models/gnn.GIN`: GIN-0, sum aggregation + sum pooling, same depth/width/head as GCN; `scripts/run_gnn.py gin` → `data/processed/predictions/gin.csv`, seeds 0–4; see ADR 2026-11-24)
- [x] MPNN using bond features (`models/gnn.GINE`: GIN + `GINEConv` with the 7-d bond features, one change vs. GIN; `scripts/run_gnn.py gine` → `data/processed/predictions/gine.csv`, seeds 0–4; see ADR 2026-11-24)
- [ ] Early stopping on validation; compared against Phase 2 (early stopping done in `train.train_gnn`; comparison waits on `evaluate.py`)

Done when: at least one GNN is trained and compared head-to-head with the baselines on the
same split.

---

## Phase 4 — Explainability Status: in progress (all three items implemented; the sweet
sanity check fails, see ADR 2026-11-25)

- [x] Atom-level attributions (integrated gradients / GNNExplainer) (`explain.integrated_gradients` / `explain_smiles`: hand-written IG on atom features, zero baseline, logit of one label, completeness gap `delta` returned; see ADR 2026-11-24)
- [x] Heatmaps on RDKit drawings (`explain.draw_attributions`: RDKit similarity map, red = towards the label, blue = away, symmetric per-molecule scale, PNG bytes; `scripts/explain_molecule.py` → `data/processed/heatmaps/`; see ADR 2026-11-24)
- [x] Sanity-checked against known bitter/sweet compounds (`explain.average_attributions` over the 5 GIN seeds + `scripts/sanity_check_attributions.py`: 8 known compounds vs. SMARTS pharmacophores, precision at k against the chance rate → `data/processed/attribution_sanity.csv`). **Bitter/umami pass, sweet fails** (saccharin/acesulfame/sucrose score 0.00; sweet attributions track molecule size, the sum-pooling signature). Stable at 512 steps, so not an IG artefact; see ADR 2026-11-25

Done when: attribution heatmaps render and pass the sanity check.

---

## Phase 5 — Demo & write-up Status: in progress

- [x] FastAPI endpoint (SMILES → predictions + attribution image) (`predict.Ensemble` over `gin_seed{0..4}` + `api.py`: `POST /predict` → mean ± seed-std per label, `validated` flag (false for salty/sour), base64 PNG of seed-averaged IG for the top or requested label, plus an `applicability_domain` block (nearest training molecule by Morgan/Tanimoto similarity, `in_domain` at 0.3; `domain.py`, ADR 2026-11-26); `uv run uvicorn tantalus.api:app`; see ADR 2026-11-25)
- [x] Simple frontend (`demo.py`: Gradio page calling `predict.Ensemble` in-process; SMILES + label to explain → canonical SMILES, probability / seed std / validated table, averaged heatmap, nearest-training-molecule / out-of-domain warning, limitations note; `uv run python -m tantalus.demo`; see ADR 2026-11-25)
- [ ] README: architecture diagram, results table, limitations (architecture diagram (Mermaid) and limitations written; results table laid out with every row "pending", waiting on the owner's `evaluate.py`)

Done when: a user can submit a SMILES and get predictions + an attribution image.
