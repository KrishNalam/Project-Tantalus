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
