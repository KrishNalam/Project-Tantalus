# Roadmap

## Phase 0 — Setup & featurization Status: in progress

- [x] CPU PyTorch installs and imports (`import torch, torch_geometric`)
- [x] Project skeleton exists (`data/{raw,processed}`, `notebooks/`, `docs/`, `src/tantalus/models/`)
- [ ] `featurize.py`: `canonicalize` / `atom_features` / `bond_features` / `smiles_to_graph` / `smiles_to_fingerprint`
- [ ] `notebooks/00_explore.ipynb` runs end-to-end
- [ ] `tests/test_featurize.py` written (by me) and reviewed
- [ ] Datasets downloaded to `data/raw/`

Done when: the featurizer works on a known molecule and the notebook demonstrates
canonicalization, atom features, fingerprints, and Tanimoto similarity.

---

## Phase 1 — Data Status: not started

- [ ] Load ChemTastesDB (primary), BitterDB, SweetenersDB
- [ ] Canonicalize, dedup across sources, resolve label conflicts, drop invalids (documented)
- [ ] Merge into one table: canonical SMILES + multi-hot labels
- [ ] Bemis–Murcko scaffold split implemented
- [ ] Class balance per split reported

Done when: a single processed table + scaffold split exists with every drop documented
and class balance reported.

---

## Phase 2 — Baselines Status: not started

- [ ] Morgan fingerprints → Random Forest
- [ ] Morgan fingerprints → XGBoost
- [ ] Scaffold split, multiple seeds, per-label ROC-AUC / PR-AUC (mean ± std)

Done when: baseline metrics are recorded on the scaffold split as the number the GNN must beat.

---

## Phase 3 — GNN Status: not started

- [ ] GCN
- [ ] GIN
- [ ] MPNN using bond features
- [ ] Early stopping on validation; compared against Phase 2

Done when: at least one GNN is trained and compared head-to-head with the baselines on the
same split.

---

## Phase 4 — Explainability Status: not started

- [ ] Atom-level attributions (integrated gradients / GNNExplainer)
- [ ] Heatmaps on RDKit drawings
- [ ] Sanity-checked against known bitter/sweet compounds

Done when: attribution heatmaps render and pass the sanity check.

---

## Phase 5 — Demo & write-up Status: not started

- [ ] FastAPI endpoint (SMILES → predictions + attribution image)
- [ ] Simple frontend
- [ ] README: architecture diagram, results table, limitations

Done when: a user can submit a SMILES and get predictions + an attribution image.
