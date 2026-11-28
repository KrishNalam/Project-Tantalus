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

### 2026-11-20 — Add BitterDB (2019 release, compound table only)

**Decision:** `scripts/download_data.py` also fetches `BitterCompoundsPropA_2019.csv` from
BitterDB's versioned `downloads/2019/` path into `data/raw/bitterdb/`, MD5-pinned like
ChemTastesDB. All requests now send a generic `User-Agent` (`tantalus-download/0.1`).
**Why:** The compound table is the only BitterDB file with SMILES (1041 rows). The receptor,
name and reference tables aren't needed for taste labels. BitterDB returns an HTML error page
(HTTP 200) to urllib's default `Python-urllib` agent. The MD5 check caught this, so the
script sets a neutral agent string. BitterDB is CC-BY-NC-ND-4.0, which is fine for a
non-commercial portfolio project, but we must not redistribute the raw or merged data.
**Rejected:** fetching all six 2019 CSVs (unused); a browser-spoofing User-Agent (a plain
client name is enough and more honest); dropping the checksum for this host (it caught the
error page, which proves the check is useful).

### 2026-11-21 — Loaders return (kept, dropped); ChemTastesDB SMILES are re-canonicalized with RDKit

**Decision:** `data.load_chemtastesdb` reads the xlsx as strings, renames columns to
snake_case, keeps the file's "canonical SMILES" column as `source_smiles`, and adds an RDKit
`canonical_smiles` column. It returns `(kept, dropped)`. `dropped` holds every removed row
plus a `drop_reason` (currently "no SMILES" for 3 footnote rows, and "SMILES not parsable
by RDKit" for 0 rows). The loader does no dedup, salt stripping, or label mapping.
**Why:** A "canonical" SMILES is only canonical within one toolkit. ChemTastesDB's column
comes from Marvin, and 3562 of 4075 strings change under RDKit. Merging on mixed toolkits'
strings would miss duplicates across sources, and missed duplicates are a leakage risk.
Returning dropped rows as data, rather than just a count or log line, makes "document
every drop" checkable and lets the Phase 1 report tabulate drops by source and reason.
`dtype=str` stops pandas turning IDs like `0001` and CAS numbers into numbers.
**Rejected:** trusting the source's canonical SMILES (cross-toolkit mismatch); silently
filtering bad rows (drops become invisible); doing dedup and label mapping inside the
loader (each needs its own decision: salts and conflicting labels, see DEVLOG 2026-11-21).

### 2026-11-21 — BitterDB loader: stereo-free SMILES, human-only rows

**Decision:** `data.load_bitterdb` reads 6 of BitterDB's 30 columns. It canonicalizes the
stereo-free `canonical_smiles` column with RDKit, not `Isomeric_smiles`, and drops the 10 rows
whose `Human` flag is 0 (bitter only to mice and other species). Like `load_chemtastesdb`,
it returns `(kept, dropped)`. Result: 1031 kept (997 unique canonical SMILES) and 10 dropped.
**Why:** ChemTastesDB's SMILES contain no stereochemistry (0 of 4075 have `@`). If one source
keeps stereo and the other doesn't, the same molecule gets two keys. With stereo-free SMILES,
965 of 997 BitterDB molecules match a ChemTastesDB entry. With isomeric SMILES, only 536 do.
Unmatched duplicates could end up on both sides of the split. The project predicts *human*
taste, so a "bitter to mice" label is the wrong target.
**Rejected:** `Isomeric_smiles` (breaks cross-source matching, and one of them fails to
parse); keeping all 30 columns (they're descriptors we can recompute, and unused); keeping
non-human rows as negatives (we have no evidence they're not bitter to humans either).

### 2026-11-21 — Scaffold key: specific Bemis–Murcko scaffold, stereo-free, "" for acyclic

**Decision:** `data.murcko_scaffold(smiles)` returns RDKit's `MurckoScaffoldSmiles` with
`includeChirality=False`. It uses the specific (atom- and bond-typed) scaffold, not the
generic carbon-skeleton one. Acyclic molecules return `""` and form one shared group.
Unparsable input returns `None`, like `canonicalize`. Salts are passed through: ring-free
counter-ions like `[Na+]` vanish from the scaffold on their own.
**Why:** The scaffold is the grouping key for the scaffold split. Molecules that share a
scaffold go to the same side, so the test set measures generalization to new chemotypes.
The specific scaffold is the standard choice (MoleculeNet / DeepChem), so results are
comparable. Ignoring chirality matches the stereo-free SMILES used for dedup. On the
3879 unique molecules of ChemTastesDB + BitterDB, there are 1005 scaffolds. 698 molecules
are acyclic (`""`), and 668 have a plain benzene scaffold.
**Rejected:** the generic scaffold (`MakeScaffoldGeneric`). It merges groups more
coarsely, which makes a harder but less standard split. It could be a later comparison.
Giving each acyclic molecule its own key would let near-identical chains (e.g. sugars
vs. sugar alcohols) leak across the split. Stripping salts first was also rejected because
the salt rule is still an open owner decision.

### 2026-11-21 — Scaffold split: Chemprop-style "balanced" split, seeded

**Decision:** `data.scaffold_split(smiles, frac=(0.8, 0.1, 0.1), seed=0)` groups rows by
`murcko_scaffold`. Groups larger than half the smaller of the val/test target sizes count as
"big" and are placed first, so they always go to train. The remaining groups are shuffled
with `seed` and assigned greedily (train while it fits, then val, then test). It returns
sorted positional indices `(train, val, test)`. It raises `ValueError` on an unparsable
SMILES or on fractions that don't sum to 1. On the 3879 unique ChemTastesDB + BitterDB
molecules: 3103 / 387 / 389, with 2 big groups (acyclic `""` and benzene), both in train.
**Why:** Every scaffold sits in exactly one split, so the test set holds unseen chemotypes.
Seeding only the small groups gives different, still scaffold-disjoint splits, which we need
for the "mean ± std over seeds" rule (test sets for seeds 0 and 1 share only 35 of ~389
molecules). Putting big groups in train first stops a single 700-molecule group from
becoming the entire test set. Duplicate SMILES share a scaffold, so they can't leak across
splits even before dedup. Raising instead of skipping bad rows keeps drops in the loaders,
where they are logged.
**Rejected:** the deterministic DeepChem split (largest groups first, no randomness). It
gives one split only, so seed variance just measures model noise, not split noise. A fully
random group order would let one huge group land in val/test. Returning boolean masks or a
split column would tie the function to a DataFrame; plain indices work for any sequence.

### 2026-11-22 — Row-level label mapping to six multi-hot labels

**Decision:** `data.assign_labels(df)` adds 0/1 columns `LABELS = [sweet, bitter, umami,
sour, salty, tasteless]` to every loader row (before dedup) and returns `(labeled, dropped)`.
BitterDB rows get `bitter = 1`. ChemTastesDB single-taste classes (Sweetness, Bitterness,
Umaminess, Sourness, Saltiness, Tastelessness) get that one label. Non-sweetness and
Non-bitterness get all zeros. Multitaste and Miscellaneous get the union of basic tastes
named in their free text (`taste_mentions`), and rows that name none are dropped (43, all
Miscellaneous). If tasteless co-occurs with a taste in one row, tasteless is set to 0.
Unknown sources or classes raise `ValueError`. On ChemTastesDB + BitterDB: 5106 rows → 5063
kept. Row-level positives: sweet 1472, bitter 2808, umami 293, sour 101, salty 48,
tasteless 233. 332 rows are all-zero (the Non-sweet/Non-bitter negatives).
**Why:** These are the owner's Phase 1 label rules (CLAUDE.md, 2026-11-22). One choice the
rules left open was made here: single-taste classes use the curated class only and ignore
cross-mentions in the free text (e.g. "bitter aftertaste" in a Sweetness row). The class is
the database's curated judgement, and keyword matching is only trusted where the class
carries no taste information. Labels are assigned per row, before dedup, so the later merge
step only has to OR rows together. Raising on unknown categories means a new data release
can't silently mislabel rows.
**Rejected:** union of mentions for every row (about 50 extra sweet/bitter positives from
keyword matching on free text that may describe aftertastes or mixtures); treating
Non-sweet/Non-bitter as tasteless (they are negatives for one taste, not "no taste");
keeping Multitaste as its own label (not a taste, and it hides the real tastes).

### 2026-11-22 — Salt rule: neutral largest-fragment parent, except salty/umami rows

**Decision:** `data.strip_salts(df)` runs after `assign_labels` and before dedup. For every
row not labeled salty or umami, `canonical_smiles` is replaced by the largest fragment
(RDKit `LargestFragmentChooser`), neutralized with `Uncharger`, re-canonicalized. The old
value is kept in `pre_salt_smiles`, and a log lists every changed row. Single-fragment
charged rows (e.g. a sulfamate anion written without its Na⁺) are neutralized too. Salty
and umami rows keep their full SMILES. Unparsable input raises. On ChemTastesDB + BitterDB
(5063 labeled rows): 849 rows lost counterions, 58 were only neutralized, 95 ion rows kept;
unique keys drop from 3839 to 3728, i.e. 111 salt/parent pairs will now merge in dedup.
**Why:** Owner rule (CLAUDE.md, Phase 1). A salt and its free acid are the same tastant for
sweet/bitter, so they must share one dedup key; otherwise near-duplicates can sit on both
sides of the split (leakage). Neutralizing also catches counterion-free charged rows, which
the rule left open: without it `[O-]` and `O` forms of one molecule would stay separate.
**Rejected:** stripping only (no neutralizing), which leaves charged parents that don't
match the neutral form; stripping salty/umami rows too (deletes the tastant, e.g. NaCl → Cl);
dropping all multi-fragment rows (loses ~850 labeled molecules).
**Known residual:** 14 salty/umami rows have a neutral parent that is also a separate row
(e.g. a sodium salt labeled salty and its acid labeled sour). They stay separate molecules.
Their Murcko scaffolds match in >99% of ion rows, so the scaffold split keeps them together.

### 2026-11-22 — Dedup and merge: OR labels per canonical SMILES, then clear tasteless

**Decision:** `data.merge_duplicates(df)` runs on `strip_salts` output and groups rows by
`canonical_smiles`. Labels are OR-ed across the group (`max` of each 0/1 column); if the
result has any real taste, `tasteless` is set to 0. The output has one row per molecule with
`n_rows` and `sources`; a merge log lists every group of 2+ rows with its distinct pre-salt
SMILES, final labels, and whether tasteless was cleared. `data.build_dataset()` runs the
whole pipeline (load → `assign_labels` → `strip_salts` → `merge_duplicates`) and writes
`molecules.csv`, `dropped.csv`, `salt_log.csv`, `merge_log.csv` to `data/processed/`
(gitignored). Result: 5063 labeled rows → 3728 molecules; 1132 merged groups (946
cross-source BitterDB+ChemTastesDB, 185 within ChemTastesDB, 1 within BitterDB), 16
tasteless labels cleared. Positives: sweet 1310, bitter 1717, umami 292, sour 98, salty 47,
tasteless 213; 306 all-zero; 228 multi-label (135 sweet+bitter).
**Why:** Owner rule (CLAUDE.md, Phase 1). In a multi-label setup, two sources saying "sweet"
and "bitter" are compatible facts, not a conflict, so OR keeps both. Dedup must happen
before the split so one molecule can't land in train and test (leakage). Writing the logs
next to the table keeps every drop and merge auditable. Free-text columns (names, notes) are
not carried into the merged table: they differ between sources and nothing downstream
needs them; they are recoverable via the logs.
**Rejected:** majority vote / source priority (throws away real secondary tastes); dropping
conflicting molecules (loses ~230 multi-taste molecules); keeping the first row's metadata
(arbitrary and order-dependent).

### 2026-11-23 — Persist scaffold splits for seeds 0–4 and their class balance

**Decision:** `data.write_splits(molecules)` runs `scaffold_split` (80/10/10) for seeds
0–4 on `molecules.csv` and writes `data/processed/splits.csv` (`canonical_smiles` +
one `seed_<k>` column of train/val/test) and `split_balance.csv` (`split_coverage`
output). `scripts/build_data.py` runs `build_dataset` then `write_splits` and prints the
minimum positives per (split, label) over seeds. Sizes per seed: 2982 / 372 / 374.
Minimum val/test positives over the 5 seeds: sweet 103/95, bitter 147/184, umami 8/9,
tasteless 10/6, sour 0/2, salty 0/0.
**Why:** Phase 2 baselines and Phase 3 GNNs must be scored on identical molecules; reading
the assignment from one file makes that true by construction instead of relying on
re-running the split with the same row order. Five seeds give the mean ± std the metrics
rule requires. The numbers confirm the owner's evaluation rule: all four headline labels
(sweet, bitter, umami, tasteless) have positives in val and test for every seed, while
salty (0) and sour (0 in some val) do not, so they stay random-split-only.
**Rejected:** recomputing splits inside each training script (silent drift if the table's
row order or the split code changes); a single seed (no variance estimate, and a lucky or
unlucky test set would go unnoticed); storing only indices (break if row order changes,
unlike keying on `canonical_smiles`).
**Caveat:** umami and tasteless have as few as 6–10 positives in a split for some seeds,
so their per-label AUCs will be noisy; report the std, not just the mean.

### 2026-11-23 — Random Forest baseline: one class-weighted forest per label, predictions saved to disk

**Decision:** `tantalus.models.baselines` loads `molecules.csv` + `splits.csv` (joined on
`canonical_smiles`), computes 2048-bit Morgan fingerprints, and fits one
`RandomForestClassifier(n_estimators=500, class_weight="balanced")` per label on the
`train` rows only (binary relevance). `scripts/run_random_forest.py` does this for every
split seed (0–4, forest `random_state` = split seed) and writes val + test probabilities and
true labels to `data/processed/predictions/random_forest.csv` (long format: one row per
(molecule, seed), `<label>_true` / `<label>_prob` columns). No metrics are computed here;
the owner's `evaluate.py` scores this file.
**Why:** One forest per label mirrors the "sigmoid per label" task: each label gets its own
trees and its own class weights, so rare labels (umami, salty) aren't drowned out by
bitter/sweet. `class_weight="balanced"` is the tree-model analogue of class-weighted BCE.
Default hyperparameters with no tuning means val and test are never used to fit anything,
so there's no leakage from model selection. Saving predictions (not scores) separates
training from evaluation, lets the owner's metric code score RF, XGBoost and GNNs the same
way, and means retraining isn't needed to add a metric.
**Rejected:** sklearn's native multi-output forest (one tree set shares splits across all
labels, and per-label class weighting is less transparent); tuning hyperparameters now (do
it on val only, after a first score exists); computing ROC-AUC inside the script (metrics
are the owner's `evaluate.py`).
**Runtime:** ~2 min for 5 seeds × 6 labels on a laptop CPU.

### 2026-11-23 — XGBoost baseline: per-label `scale_pos_weight`, CPU-only wheel, shared runner script

**Decision:** `baselines.train_xgboost` fits one `XGBClassifier(n_estimators=500,
tree_method="hist")` per label on `train` rows only, with
`scale_pos_weight = n_negative / n_positive` computed per label and otherwise default
hyperparameters (`random_state` = split seed). A label with a single class in train gets
`None` and a constant-0 prediction (never happens on the current splits; salty has ≥ 44
train positives). The dependency is `xgboost-cpu`, not `xgboost`. The runner
`scripts/run_random_forest.py` was renamed to `scripts/run_baseline.py <model>` and writes
`data/processed/predictions/<model>.csv` in the same format as the RF file.
**Why:** `scale_pos_weight` is XGBoost's built-in way to up-weight positives; with
n_neg/n_pos it gives positives and negatives equal total weight, the same balance as the
RF's `class_weight="balanced"` and the class-weighted BCE planned for the GNN, so the
baselines differ in model family, not in how imbalance is handled. Default hyperparameters
keep val/test out of fitting (no tuning leakage). The plain `xgboost` wheel pulled in
`nvidia-nccl-cu13` (a CUDA library); the project is CPU-only, and `xgboost-cpu` is the same
library without GPU dependencies. One runner script for both models removes a duplicated
training loop and guarantees both prediction files have the same columns.
**Rejected:** `sample_weight` per row (equivalent here, more code); a native multi-label
XGBoost model (shares trees across labels, same objection as the multi-output forest);
early stopping on val (that would be tuning; do it later, on val only, for all models
alike); a separate `run_xgboost.py` (copy of the RF loop).
**Runtime:** ~1m45s for 5 seeds × 6 labels on a 4-core laptop CPU.

### 2026-11-23 — Random split saved alongside the scaffold split (deliberate comparison only)

**Decision:** `data.random_split` shuffles row positions with `random.Random(seed)` and cuts
them 80/10/10. `data.write_splits(..., method="random")` saves it for seeds 0–4 to
`data/processed/random_splits.csv` + `random_split_balance.csv`. These use the same format
as the scaffold files, which are byte-identical to before. `scripts/build_data.py` writes
both. `scripts/run_baseline.py <model> --split random` writes
`predictions/<model>_random.csv`. The default is still the scaffold split.
`write_splits` now computes the class balance from the assignments it just made instead of
re-running `scaffold_split` through `split_coverage`.
**Why:** The owner's Phase 1 rule says salty and sour are reported only on a random split,
clearly flagged as a comparison, because the scaffold split leaves them with 0 val/test
positives. A random split over the deduplicated table gives min val/test positives of
salty 3/1 and sour 8/9 over seeds 0–4. It also gives the usual random-vs-scaffold gap, which
shows how much the scaffold split's "new chemistry" test costs. Saving it to disk is the
same choice as for the scaffold split: every model reads identical molecules. Separate
files and a `_random` suffix on predictions mean a random-split number can't be mistaken
for a headline number.
**Leakage note:** a random split is safe from *exact* duplicates only because the
dedup/merge runs first (one row per canonical SMILES). Close analogues still land on both
sides. That's the optimism the comparison is meant to show, so its numbers must never be
headlined.
**Rejected:** stratified / iterative multi-label stratification (more code and a new
dependency for a side comparison; would also make the random split less "plain random");
an extra `split_type` column in `splits.csv` (mixes headline and comparison splits in one
file and breaks the existing loaders); re-splitting inside the training scripts (same drift
risk as for the scaffold split).
**Caveat:** salty has only 1 test positive for seeds 2 and 4, so its random-split ROC-AUC is
close to meaningless per seed. Report it with its std and the positive count, or pool
val+test for salty.

### 2026-11-23 — GCN architecture (first Phase 3 model)

**Decision:** `models/gnn.py` defines `GCN`: 3 × (`GCNConv` → ReLU → dropout 0.2) with hidden
size 128 over the 33-d atom features, `global_mean_pool` per molecule, then an MLP head
(Linear 128 → ReLU → dropout → Linear 6) that returns **logits**, one per label. No sigmoid
inside the model. ~55k parameters. Bond features (`edge_attr`) are ignored on purpose.
**Why:** GCN is the simplest message-passing model, so it is the reference point that GIN
and the bond-aware MPNN must improve on. Returning logits lets training use
`BCEWithLogitsLoss` (sigmoid + BCE fused, numerically stable, with a per-label `pos_weight`
for class weighting). That is still the "sigmoid per label, not softmax" task from
CLAUDE.md. 3 layers is the low end of the planned 3–5, which roughly matches the radius-2/3
context of the Morgan baseline. Mean pooling keeps the molecule embedding on the same scale
for small and large molecules. 128 hidden units is small enough to train on a CPU over ~3k
graphs. Hyperparameters are fixed defaults, not tuned. Any tuning later uses val only.
**Rejected:** sum pooling (GIN's choice, more expressive, kept for GIN so the two models
differ in a way that can be explained); max pooling; BatchNorm/residual connections (not
needed at 3 layers, can be added if deeper models degrade); a sigmoid output layer with
`BCELoss` (less stable, and `pos_weight` is only on `BCEWithLogitsLoss`).

### 2026-11-23 — GNN training loop (`train.py`)

**Decision:** `train.train_gnn` trains with Adam (lr 1e-3, batch 64) on
`BCEWithLogitsLoss(pos_weight = n_neg / n_pos per label)`, computed from the **train split
only**. After each epoch it computes the val loss with the same weighting. It stops after
20 epochs without improvement (cap 200) and restores the best-val weights.
`scripts/run_gnn.py <model>` runs seeds 0–4 on the saved splits and writes val/test
probabilities in the same format as the baselines (`predictions/<model>.csv`). Molecules and
splits are loaded via the new shared `data.load_split_table`, which the baselines use too.
**Why:** `pos_weight` is the BCE equivalent of the baselines' `class_weight="balanced"` /
`scale_pos_weight`, so all models handle imbalance the same way. Computing it on train only
keeps held-out labels out of the loss. Early stopping on val loss is the cheapest
regulariser and needs no tuning. Restoring the best weights (not the last) means the model
we report is the one val picked. A shared loader and output format let `evaluate.py` score
every model identically.
**Leakage note:** val now selects the stopping epoch, so **val predictions are optimistically
biased for the GNN**. Headline numbers must come from **test**.
**Rejected:** early stopping on val ROC-AUC (noisy with 8–10 umami/tasteless positives, and
the loss is smooth); a learning-rate scheduler or tuning (no tuning before the baseline
comparison); a weighted sampler (a second imbalance mechanism on top of `pos_weight`).

### 2026-11-24 — GIN architecture (second Phase 3 model)

**Decision:** `models/gnn.py` adds `GIN`: 3 × (`GINConv` → ReLU → dropout 0.2), where each
`GINConv` wraps a 2-layer MLP (Linear → ReLU → Linear, width 128) and uses a fixed `eps = 0`
("GIN-0"). Atoms are combined with `global_add_pool` (sum), then the same MLP head as `GCN`
returns 6 logits. ~104k parameters. Bond features are still ignored. Trained with the
unchanged `train.train_gnn` via `scripts/run_gnn.py gin` → `predictions/gin.csv`.
**Why:** GIN changes exactly two things relative to GCN: sum aggregation (no degree
normalisation) followed by an MLP, and sum pooling. Sum + an injective MLP is as
discriminative as the Weisfeiler–Lehman test, while GCN's normalised mean can't tell some
different neighbourhoods apart. Sum pooling keeps counts (two hydroxyls vs. four), which
can matter for taste (e.g. polyols and sweetness). Depth, width, dropout, head, optimiser
and early stopping all match GCN, so any difference can be explained by aggregation and
pooling.
**Rejected:** learnable `eps` (the GIN paper found GIN-0 comparable, and it's one less
parameter to explain); BatchNorm inside the MLP (the paper uses it, but GCN has none, and
adding it would be a second change at once); mean pooling (would undo half the point of
GIN); bond features via `GINEConv` (that is the MPNN step).

### 2026-11-24 — Bond-aware MPNN: GINE (third Phase 3 model)

**Decision:** `models/gnn.py` adds `GINE`, a subclass of `GIN` that swaps each `GINConv`
for a `GINEConv(edge_dim=7)`. A neighbour's message becomes `ReLU(h_j + W·e_ji)`, where
`e_ji` is the 7-d bond feature vector (type, conjugated, in ring) and `W` is a per-layer
learned linear projection. Everything else (3 × 128, GIN-0, MLPs, dropout 0.2, sum pooling,
head, `train_gnn`) is inherited unchanged. ~107k parameters (+~2k vs. GIN). Trained via
`scripts/run_gnn.py gine` → `predictions/gine.csv`, seeds 0–4.
**Why:** It is the smallest step from GIN that uses bond features: exactly one change, so a
GIN/GINE gap can be attributed to bond information. Bond type matters chemically (C=O vs.
C–O; aromatic vs. aliphatic), and GCN/GIN only see it indirectly via atom features such as
hybridisation and aromaticity.
**Rejected:** `NNConv` (edge-conditioned convolution, the original Gilmer MPNN: an MLP maps
each bond to a full `d × d` weight matrix, ~128² params per bond-MLP output, heavier and
slower on CPU, and changes more than one thing vs. GIN); a GRU update / Set2Set readout as
in Gilmer et al. (more moving parts before any scored comparison); concatenating bond
features into atom features (loses which neighbour the bond leads to).

### 2026-11-24 — Save per-seed GNN checkpoints for Phase 4

**Decision:** `scripts/run_gnn.py` saves each seed's best-val model to
`data/processed/checkpoints/<model>[_random]_seed<k>.pt` as a dict `{model, split, seed,
best_epoch, state_dict}`. `models/gnn.load_checkpoint(path)` rebuilds the model from the
`MODELS` registry (moved from the script into `models/gnn.py`) with default hyperparameters,
loads the weights with `torch.load(..., weights_only=True)` and returns `(model, metadata)`
in eval mode. Checkpoints are gitignored (`data/`, `*.pt`). GIN was retrained to produce
them.
**Why:** Phase 4 attributions must explain the *same* models whose predictions
`evaluate.py` scores. Retraining inside `explain.py` would not give the same weights,
because CPU training here is not bit-reproducible (see Known issues in STATUS.md). Saving a
`state_dict` plus the model name (not the pickled module) keeps the files independent of
class internals, and `weights_only=True` refuses to unpickle arbitrary objects.
**Rejected:** `torch.save(model)` (pickles the class by import path, so a refactor breaks old
files, and loading needs full unpickling); storing hyperparameters in the checkpoint (every
model uses defaults today, so it would be untested code; add it when a non-default model is
trained); retraining in Phase 4 (slow and not identical to the scored models).

### 2026-11-24 — Hand-written integrated gradients for atom attributions

**Decision:** `src/tantalus/explain.py` adds `integrated_gradients(model, graph, label,
steps=128)` and `explain_smiles(...)`. The baseline sets all atom features to zero. Bonds,
`edge_attr` and connectivity stay fixed. The method attributes the chosen label's
**logit**, approximates the path integral with the midpoint rule, and runs all `steps`
scaled copies as a single PyG batch. Each atom's score is the sum over its features of
`(x - baseline) * mean_gradient`. The function also returns the completeness gap `delta`.
**Why:** IG satisfies *completeness*: atom scores add up to `logit(x) - logit(baseline)`.
That gives a numeric check (`delta`) that the attribution is right, which GNNExplainer
doesn't offer. The hand-written version is about 20 lines, adds no dependency, and every
line can be explained. Using logits avoids sigmoid saturation, since GIN gives some
probabilities of exactly 0/1, where gradients vanish. Batching the path means one
forward/backward pass: about 0.02 s per molecule on CPU.
**Rejected:** Captum (an extra dependency, and its PyG wrapper hides the batching that
matters here); GNNExplainer (it learns a mask by optimisation, so results are stochastic
and have no completeness check); attributing probabilities (saturation); attributing bond
features too (the GIN checkpoints don't use them; can be added for GINE later).

### 2026-11-24 — Attribution heatmaps with RDKit similarity maps

**Decision:** `explain.draw_attributions(smiles, attributions, size=(400, 300))` returns PNG
bytes (or `None` for an invalid SMILES). It draws RDKit's similarity map
(`SimilarityMaps.GetSimilarityMapFromWeights` on a `MolDraw2DCairo`). Each atom adds a
Gaussian blob weighted by its IG score, using the diverging `bwr` colormap: red pushes towards
the label, blue pushes away, and white means no effect. The colour scale is symmetric around 0
and set by the molecule's largest absolute score. A one-atom molecule (similarity maps need
two atoms) is drawn with a plain red/blue atom highlight. `scripts/explain_molecule.py
"<SMILES>" <label> [--model gin --seed 0]` loads a saved checkpoint and writes
`data/processed/heatmaps/<model>_seed<k>_<label>_<md5[:8]>.png` (gitignored).
**Why:** Similarity maps are RDKit's standard look for per-atom contributions, so there is no new
dependency. A diverging map with zero fixed at white is the right encoding for signed scores:
a sequential map would make negative atoms look like "weak positives". Returning bytes rather
than writing a file means the Phase 5 FastAPI endpoint can send the image directly. Atom
order matches because RDKit keeps the input order when the same SMILES string is parsed again,
and `smiles_to_graph` doesn't reorder atoms. A length mismatch raises instead of silently
mis-colouring.
**Rejected:** a fixed colour scale across molecules (IG sums differ by an order of magnitude
between molecules, so small molecules would wash out; per-molecule scaling is stated in the
docstring, and `delta`/the sum are printed alongside); per-atom highlight circles for every
molecule (simpler, but a smoothed map reads better for substructures); matplotlib figures
(heavier, and the result isn't PNG bytes for the API); naming files by SMILES (contains `/`, `\`,
`#`).

### 2026-11-25 — Attribution sanity check against known taste pharmacophores

**Decision:** `explain.average_attributions(models, smiles, label, steps)` averages integrated
gradients over several checkpoints and returns `(mean, std, max_abs_delta)` per atom, with a
population std (so one model gives 0, not `nan`). `scripts/sanity_check_attributions.py` runs
it over the five GIN seeds for a fixed panel of eight compounds whose taste chemistry is known
(caffeine, quinine, denatonium / saccharin, acesulfame, aspartame, sucrose / MSG). Each entry
carries a tuple of SMARTS for the motif the literature holds responsible, because a
pharmacophore is often several disconnected groups (aspartame's free amine *and* its
carboxyl). The check is quantitative: take the top-|motif| atoms by mean attribution and
report *precision at k* against the chance rate |motif| / |atoms|, plus the share of positive
attribution mass on the motif, the mean predicted probability, the max across-seed std, and
the worst completeness gap. Output: `data/processed/attribution_sanity.csv` and one
mean-attribution heatmap per molecule (gitignored).
**Result (this is a partial failure, recorded deliberately):** bitter and umami pass —
denatonium's quaternary ammonium is the single top atom (chance 0.03), MSG scores 0.83 at k
against chance 0.55, caffeine and quinine beat chance modestly. **Sweet fails**: saccharin,
acesulfame and sucrose all score 0.00 precision at k. The top sweet atoms are carbons at a
near-uniform +4 each, so the attribution is tracking molecule size rather than chemistry —
the expected signature of sum pooling, which already saturates GIN's probabilities (see the
GIN ADR and STATUS known issues). Across-seed std is large everywhere (up to 16.8 for
denatonium), so single-seed heatmaps should not be trusted on their own. Conclusions are
stable at 512 steps, where every completeness gap is under 0.45, so this is model behaviour
and not IG approximation error.
**Why this design:** averaging over seeds is what separates "this run looked at that atom"
from "the architecture looks at that atom", and the std makes the disagreement visible instead
of hiding it in a mean. Precision at k with the chance rate printed next to it keeps a motif
that covers half the molecule (MSG: 6 of 11 atoms) from looking like a success by default.
Reporting the predicted probability matters because an attribution for a label the model
doesn't predict explains a *negative* answer: saccharin and acesulfame sit at about 0.51, so
the model is barely calling them sweet at all.
**Rejected:** declaring the check passed on the bitter cases alone (the sweet result is the
interesting one and belongs in the write-up); a pass/fail threshold (eight hand-picked
molecules can't support one, and the motifs are approximate); putting this in `tests/` (it is
the owner's area, and a negative result shouldn't break the suite); mean pooling to fix the
size artefact (that changes Phase 3's architecture and must be a measured comparison, not a
fix smuggled in through Phase 4).

### 2026-11-25 — FastAPI demo endpoint over a seed ensemble

**Decision:** Phase 5's endpoint is `POST /predict` in `src/tantalus/api.py`, a thin HTTP
wrapper around `predict.Ensemble`, which loads the five `gin_seed{0..4}.pt` checkpoints once at
startup (FastAPI `lifespan`). For a SMILES it canonicalizes first (as in training), returns the
mean probability and the across-seed std for every label, a `validated` flag that is false for
salty and sour, and a base64 PNG of the seed-averaged integrated-gradients heatmap for one
label (the requested one, or the highest-probability label by default). Invalid SMILES and
unknown labels return HTTP 422; SMILES are capped at 500 characters.
**Why:** Phase 4 showed single-seed attributions disagree a lot, so the demo must not show one
seed's heatmap; averaging the probabilities over the same five models keeps prediction and
explanation consistent, and the std is an honest uncertainty signal for free. GIN because it
has checkpoints and the best informal pooled test AUCs; swapping in another architecture is one
argument. Keeping the ensemble framework-free means the frontend (Gradio or a notebook) can call
it without HTTP. The canonical string is used for both the graph and the drawing so atom order
lines up. The endpoint is a plain `def` so FastAPI runs CPU-bound inference in a thread instead
of blocking the event loop. Salty/sour are still predicted (the model has those heads) but
flagged, because they were only ever scored on the random comparison split.
**Rejected:** a separate `GET /explain` returning `image/png` (two round trips and the client
must keep them in sync; base64 in JSON costs ~33% size on a ~20 KB image); one checkpoint
(cheaper but shows a seed-specific heatmap); defaulting the explained label to the top
*headline* label (it explained NaCl as "bitter" although salty was 1.00 — confusing in a demo);
hiding salty/sour entirely; serving data from `data/` (BitterDB is CC-BY-NC-ND, the API only
serves model outputs).

### 2026-11-25 — Gradio frontend calling the ensemble in-process

**Decision:** Phase 5's frontend is a Gradio page, `src/tantalus/demo.py` (`uv run python -m
tantalus.demo`, port 7860). It calls `predict.Ensemble` directly, not the FastAPI endpoint. One
SMILES box and a "label to explain" dropdown (default: top label) produce the canonical SMILES,
a table of mean probability, seed std and a `validated` column (no for salty/sour), the
seed-averaged heatmap with its completeness gap, and a fixed note on how to read the output and
its known limits (no out-of-domain check, sweet heatmaps track size). Empty, over-500-character
and unparseable inputs are shown as `gr.Error` messages in the page. Checkpoints load lazily on
the first request through `functools.cache`.
**Why:** Gradio gives a working page in about 100 lines of Python, with nothing new to learn
for the stack, and it can be deployed as-is to Hugging Face Spaces. Calling the ensemble
in-process means one command runs the demo, and the outputs are identical to the API because
both share `Ensemble`. The page states the limitations next to the numbers, so a viewer doesn't
take a confident probability on an odd molecule at face value.
**Rejected:** a Next.js page (a second language and toolchain for a portfolio demo; it would
also need the API running and CORS); Gradio calling the FastAPI endpoint over HTTP (two
processes to start for no gain locally); mounting Gradio inside the FastAPI app (couples the
two; worth revisiting if both have to be deployed as one service); loading checkpoints at
import time (makes `import tantalus.demo` slow and fail without checkpoints).

### 2026-11-26 — Pin GNN training to one CPU thread for reproducibility

**Decision:** `train.make_reproducible(seed, num_threads=1)` seeds torch, pins intra-op threads
with `torch.set_num_threads(1)` and turns on `torch.use_deterministic_algorithms(True)`.
`scripts/run_gnn.py` calls it before building each seed's model (it replaces the bare
`torch.manual_seed`).
**Why:** Measured, not guessed: GIN seed 0 trained for 15 epochs gave identical weights for two
runs at the same thread count (also across processes), but 1, 2 and 4 threads gave weights up
to 0.42 apart and best epochs 15/13/9. PyTorch splits reductions (the scatter-add in message
passing, matmul) into one chunk per thread, and float addition is not associative, so the
thread count changes the rounding. One thread is the only setting that gives the same answer on
any machine. The deterministic-algorithms flag costs nothing on CPU today (GCN, GIN and GINE all
train with it on) and will raise loudly if a future op has no deterministic kernel.
**Cost:** about 25% slower than the default 2 threads on this laptop (25 s vs 20 s for 15 GIN
epochs), so roughly 7-8 min instead of 6 for five seeds.
**Not retrained:** the existing `gin_seed{0..4}.pt` and `predictions/*.csv` were made with the
default thread count and stay the source of truth for the numbers already recorded. A rerun of
`run_gnn.py` will now reproduce itself exactly, but will differ slightly from those files.
**Rejected:** pinning to 2 threads (faster here, but the result then depends on the thread count
another machine would choose if someone removes the pin, and it's no more portable);
`use_deterministic_algorithms` alone (it guards against nondeterministic kernels, mostly a CUDA
concern, but does not fix how a reduction is split across threads; not tested on its own here); an env var such as
`OMP_NUM_THREADS` (easy to forget, and it lives outside the code).

### 2026-11-26 — Flag out-of-domain molecules by nearest-neighbour Tanimoto similarity

**Decision:** `domain.py` adds `ApplicabilityDomain`, built from `data/processed/molecules.csv`:
it Morgan-fingerprints all 3728 molecules once and, for a query, returns the most similar
reference molecule plus a boolean `in_domain` at a **0.3** Tanimoto threshold. `predict.Ensemble`
builds one at construction and every `Prediction` carries the check; the API returns it as
`applicability_domain` and the Gradio page prints the nearest training molecule, the similarity
and a warning when the query is outside.
**Why:** the demo's worst failure was silent: methane gets salty 0.79 with a small seed std,
because all five models are equally sure about a molecule none of them has seen anything like.
The seed spread measures *model* disagreement, not *data* coverage, so it cannot catch this. A
nearest-neighbour fingerprint-similarity check is the standard, cheap applicability-domain test
and needs no new dependency: the fingerprints are already in `featurize.py`.
**Threshold, measured not assumed:** scaffold-test molecules from seed 0 reach a median 0.62
similarity to their train split, 5th percentile 0.27. At 0.3 the flag fires on ~7% of molecules
the model is actually evaluated on, while methane (0.04) and cisplatin (0.11) are flagged. 0.4
(the common literature default) flagged 18% of legitimate test molecules, which would train
users to ignore the warning.
**Reference set is the whole merged table**, not one seed's train split: the demo ensembles five
models whose scaffold splits differ, so a molecule held out by one seed was trained on by the
other four. The union is the chemical space the ensemble as a whole has seen. (For scoring a
single seed, the correct reference would be that seed's train split only.)
**Cost:** about 4 s to fingerprint the reference set at startup, then one 3728x2048 matrix
product (~1 ms) per query. Implemented as a matrix product rather than RDKit's
`BulkTanimotoSimilarity` to stay in numpy and avoid converting bit vectors back to RDKit objects.
**Behaviour unchanged otherwise:** the probability is still returned as-is, not suppressed or
widened. Flagging is honest; silently withholding an answer would hide which molecules are hard.
**Rejected:** a bounding box / PCA-distance domain on descriptors (another featurization to
justify, and worse on sparse binary fingerprints); mean similarity to the k nearest neighbours
(smoother, but one near-duplicate is exactly the evidence we want, and max is easier to explain);
conformal prediction or an explicit uncertainty model (the principled answer for calibrated
intervals, but a project of its own and it needs the scoring code first); training a one-class
novelty detector (a second model to validate for a check the fingerprints already give).

### 2026-11-26 — Measure held-out similarity to each seed's own train split

**Decision:** `domain.split_similarity` + `scripts/report_split_similarity.py` compute, for each
split method (scaffold, random) and seed 0–4, the nearest-neighbour Morgan/Tanimoto similarity
of every val/test molecule to **that seed's train molecules only**, and save median / p05 / p25 /
mean / fraction below the 0.3 out-of-domain threshold to `data/processed/split_similarity.csv`.
Result (test, mean over seeds): scaffold median 0.58 (p05 0.28, 7.3% below 0.3); random median
0.72 (p05 0.34, 4.0% below 0.3).
**Why:** the claim "the scaffold split is harder than a random split" was so far only asserted.
This puts a number on it, so random-split scores (the only place salty/sour are scored) can be
read with the right discount, and it re-checks the 0.3 threshold on all five seeds instead of
seed 0 only (5–9% of scaffold-test molecules flagged on every seed).
**Reference = own train split, not `molecules.csv`:** comparing against the whole table would
match each molecule against itself (similarity 1.0) and against other held-out molecules, which
says nothing about what the model trained on. This differs on purpose from the demo's
`ApplicabilityDomain` (whole table, because the 5-seed ensemble has collectively seen it).
**Rejected:** a per-label breakdown (umami/tasteless have 6–10 test positives, too few for a
stable percentile); a histogram plot (a table is enough to quote and diff; plots belong in a
notebook if wanted).

### 2026-11-26 — Regenerate GCN/GINE under one-thread training; keep the GIN checkpoints

**Decision:** reran `scripts/run_gnn.py gcn` and `gine` under `train.make_reproducible`, which
replaced `predictions/{gcn,gine}.csv` and created `checkpoints/{gcn,gine}_seed{0..4}.pt` (none
existed). Reloading seed 0 of each reproduces its CSV to < 2e-7. The five GIN checkpoints and
`gin.csv` were **not** regenerated.
**Why:** a comparison table should only contain runs that can be reproduced, and Phase 4 code
(IG, heatmaps) can now run on all three architectures. The rerun also measured how big
thread-count noise is: pooled test ROC-AUC (informal check, not the official metric) moved
GCN 0.835/0.835/0.946/0.805 → 0.838/0.835/0.947/0.808 (sweet/bitter/umami/tasteless; same best
epochs) but GINE 0.81/0.83/0.935/0.847 → 0.836/0.85/0.932/0.849 (best epochs changed). A
~0.025 shift from thread count alone is as large as the GIN-vs-GINE gap reported earlier, so the
earlier remark "bond features did not help sweet/bitter" was within noise. Only per-seed mean ±
std from `evaluate.py` can settle it.
**Why GIN is kept as is:** the demo, the API and the Phase 4 sanity-check numbers (ADR
2026-11-25) are all quoted from `gin_seed{0..4}.pt`; replacing them would silently change
published numbers. It stays the one pre-fix model, documented in STATUS/README.
**Rejected:** rerunning GIN too (cleaner, but invalidates every quoted Phase 4/5 number for a
difference at noise level; worth doing together with the mean-pooling GIN experiment, when the
sanity check is rerun anyway); leaving GCN/GINE without checkpoints (they couldn't be explained
or reloaded).

### 2026-11-26 — Mean-pooling GIN as a controlled test of the sweet size artefact

**Decision:** added `pooling` (`"sum"` default | `"mean"`) to `models/gnn.GIN` (GINE inherits it)
and a `GINMean` subclass registered as `gin_mean`, which differs from GIN **only** in the
readout. Trained it with `run_gnn.py gin_mean` (one thread, seeds 0–4 scaffold splits →
`predictions/gin_mean.csv`, `checkpoints/gin_mean_seed{0..4}.pt`) and reran
`sanity_check_attributions.py --model gin_mean`, which now writes
`attribution_sanity_<model>.csv` for non-GIN models so the GIN numbers aren't overwritten.
Refactor check: `gin_seed0.pt` still reproduces `gin.csv` (diff 1e-9).
**Result:** the size artefact is the readout. Sweet IG score sums fall from 39/30/62/74
(saccharin/acesulfame/aspartame/sucrose, GIN) to 7.2/6.6/8.1/8.8. Precision at k: saccharin
0.00 → 0.50 (chance 0.33), acesulfame 0.00 → 0.50 (0.40), aspartame 0.25 → 0.25 (0.19), sucrose
0.00 → 0.00 (0.35). Bitter/umami still pass (denatonium 1.00, MSG 0.83, caffeine 0.50, quinine
0.25, same as GIN). So sweet moves from "fails" to "weakly above chance on 3 of 4", not a pass.
Saturated (exactly 0/1) test probabilities fall from 0.29% to 0.08%.
**Caveat:** seed 4 stopped at best epoch 3 (sweet test AUC 0.59, bitter 0.62 vs. 0.79–0.89 for
seeds 0–3), so the informal per-seed mean sweet test ROC-AUC is 0.80 ± 0.12 vs. GIN 0.86 ± 0.05.
An early val-loss minimum plus patience 20 can lock in an undertrained model; that seed is also
in the attribution average. Not acted on here (changing patience/min-epochs would change every
model's protocol).
**Why not switch the demo:** which GIN to serve is an accuracy question that needs the owner's
`evaluate.py`; the demo, API and the 2026-11-25 numbers stay on `gin_seed{0..4}`.
**Rejected:** a `pooling` CLI flag on `run_gnn.py` (checkpoints store only the model name, so
`load_checkpoint` couldn't rebuild a flagged variant; a registered subclass round-trips for
free); attention/set2set pooling (more than one change at once, so a result couldn't be pinned
on removing size); dropping seed 4 (cherry-picking).

### 2026-11-26 — Opt-in minimum-epoch guard for early stopping

**Decision:** `train.train_gnn` takes `min_epochs` (default 1 = plain early stopping). Epochs
before `min_epochs` are trained and still get a val pass, but they can't become the best epoch
and can't trigger stopping. `run_gnn.py --min-epochs N` turns it on, stores `min_epochs` in each
checkpoint, and appends `_min<N>` to the output names. Ran `run_gnn.py gin_mean --min-epochs 20`
→ `predictions/gin_mean_min20.csv`, `checkpoints/gin_mean_min20_seed{0..4}.pt`.
N = 20 was set to equal `patience` before looking at any scores. It wasn't tuned.
**Result:** seeds 0–3 are bit-identical to `gin_mean` (best epochs 30/84/48/49, max prediction
diff 0.0), so the guard only changes which epoch gets picked. Seed 4 moves from best epoch 3 to 24.
Informal per-seed test ROC-AUC (throwaway script, not the official metric): sweet 0.80 ± 0.11
→ 0.84 ± 0.04, bitter 0.81 ± 0.10 → 0.85 ± 0.04, umami 0.91 → 0.93, tasteless 0.82 → 0.83.
Seed 4 sweet goes 0.59 → 0.82.
**Implementation trap found:** the first version skipped the val pass during warm-up, and seeds
0–3 changed too (best epochs 28/34/52/51). Iterating any `DataLoader` draws a base seed from
torch's global RNG, so dropping val passes shifted every later dropout mask. The val pass now
always runs, and the guard gates only the selection.
**Why opt-in, not the new default:** turning it on everywhere changes the protocol for every
GNN. GIN has a seed whose best epoch is 14, so `gin_seed{0..4}.pt` (which the demo, API and
quoted sanity-check numbers use) would no longer be reproducible from the script. Making it the
default is best done together with the planned one-thread GIN regeneration, once `evaluate.py`
exists.
**Rejected:** dropping seed 4 (cherry-picking); a larger `patience` (it only lets training run
longer *after* the early minimum, and the epoch-3 weights would still win if val never beats
them); choosing N on test AUC (that would make test a tuning set); skipping val passes during
warm-up (changes the training trajectory, see above).

### 2026-11-27 — Regenerate GIN under one-thread training, guard off

**Decision:** carried out the rerun deferred by the ADR above: `run_gnn.py gin` under
`train.make_reproducible`, replacing `predictions/gin.csv` and `checkpoints/gin_seed{0..4}.pt`,
then reran `sanity_check_attributions.py` on them. The minimum-epoch guard stays off
(`min_epochs=1`), the same protocol as GCN and GINE. The old two-thread outputs are kept
locally in `data/processed/legacy_2thread/` (gitignored).
**Why:** GIN was the only GNN left on the old thread-dependent setup, so the comparison
`evaluate.py` will produce would have mixed two protocols and a 0.02 architecture gap would be
unreadable. Confounder removed before the numbers exist rather than caveated after.
**Why guard off:** turning it on for GIN alone would give one model a different stopping rule,
and it would be a no-op anyway — new best epochs 29/50/61/46/47 are all past 20.
**Effect:** informal per-seed test ROC-AUC (not the official metric) moved by 0.01–0.03, the
size of the thread-count noise already measured on GINE, so no architecture conclusion changes.
The sweet attribution failure of ADR 2026-11-25 reproduces (all four sweet cases at or below
chance, score sums 32–78); bitter and umami still pass. Numbers in `DEVLOG.md` 2026-11-27.
**Rejected:** deleting the two-thread outputs (they back previously quoted numbers, so keeping
them shows the before/after); leaving GIN as the one pre-fix model (the reason to keep it —
protecting already-published Phase 4/5 numbers — expired once those numbers were shown to be
reproducible under the new protocol).
