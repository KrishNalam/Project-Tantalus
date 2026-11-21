# Decision log (ADRs)

Architecture Decision Records: short, dated notes on **what was decided and why**, plus
the alternatives rejected. Maintained by Claude, and an entry is only added **after** the
decision is actually implemented in the repo — never ahead of the code.

Format per entry: `### YYYY-MM-DD — <title>` then **Decision / Why / Rejected**.

---

### 2026-09-30 — Use uv for dependency management and packaging

**Decision:** Manage the project with [uv] (Python 3.12 pinned via `.python-version`);
add deps with `uv add`, run with `uv run`. Build backend is `uv_build`.
**Why:** Fast, reproducible installs with a committed `uv.lock`; single tool for envs,
deps, and running.
**Rejected:** plain `pip`/`venv` (no lockfile by default), Poetry/conda (heavier).

### 2026-09-30 — Install PyTorch from the CPU-only wheel index

**Decision:** Add `torch` explicitly and route it to PyTorch's CPU wheel index via
`[[tool.uv.index]]` (`name = "pytorch-cpu"`, `explicit = true`) + `[tool.uv.sources]`
(`torch = { index = "pytorch-cpu" }`). Result: `torch==2.14.1+cpu`.
**Why:** PyTorch Geometric doesn't depend on `torch` directly (it can't know the desired
CPU/CUDA build), so torch must be added explicitly. On Linux x86, default PyPI `torch`
pulls the CUDA build (~2–3 GB) — useless on this CPU-only laptop. `explicit = true` keeps
the CPU index scoped to torch only, so it doesn't shadow PyPI for other packages.
**Rejected:** default PyPI torch (drags in CUDA); a `torch==…+cpu` local-version pin
(fiddly with uv's resolver).

### 2026-11-19 — Atom/bond feature vector layout (33 + 7 dims)

**Decision:** `atom_features` concatenates one-hots (each with a trailing "other" slot) for
element (10), degree (7), formal charge (4), hybridization (4), total H count (6), then two
0/1 flags for aromatic and in-ring → 33 dims. `bond_features` is bond-type one-hot with an
"other" slot (5) plus conjugated and in-ring flags → 7 dims. Widths are exported as
`ATOM_FEATURE_DIM` / `BOND_FEATURE_DIM`, computed from the choice lists. `smiles_to_graph`
stores each bond in both directions with duplicated `edge_attr` rows, and keeps the
canonical SMILES on the `Data` object.
**Why:** One-hot avoids implying a false ordering (e.g. "N > C"); the "other" slot means
rare elements/charges/bond types (dative, zwitterions) still get a valid vector instead of
crashing. Booleans need only one dimension. Deriving dims from the lists keeps model
input sizes in sync if a choice list changes.
**Rejected:** integer-coded categories (fake ordinal structure); a bond-type one-hot
without "other" (would silently encode unusual bonds as all-zeros); two-slot one-hots for
booleans (redundant).

### 2026-11-20 — Pin raw datasets to a fixed version and verify by checksum

**Decision:** `scripts/download_data.py` fetches ChemTastesDB from a fixed Zenodo record
(`15051366`, v2.1, CC-BY-4.0) and checks each file against a hard-coded MD5. It downloads
to a `.part` file and only renames it after the checksum matches. Files already present with
the right checksum are skipped. Only the database/references `.xlsx` and the readme are
fetched. The `.rar` of per-molecule structure files is skipped, because the SMILES are in the xlsx.
**Why:** Zenodo's "latest version" link moves (v1.0 → v2.1 already). Pinning the record keeps
results reproducible, and the checksum catches truncated or altered downloads. Raw data stays
out of git (`data/` is gitignored), so the script is the committed recipe for rebuilding it.
**Rejected:** following the "latest" redirect (silent dataset drift between runs); committing
the raw files (licence/size, and CLAUDE.md forbids committing `data/`); a manual download
README with no checks (not reproducible).
