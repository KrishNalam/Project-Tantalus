"""SMILES --> canonical SMILES, molecular graphs and Morgan fingerprints"""

import numpy as np  # fingerprint returned as numpy array
import torch  # turns feature arrays into torch tensors
from rdkit import Chem  # molecule API
from rdkit.Chem import rdFingerprintGenerator  # building Morgan fingerprints
from torch_geometric.data import Data

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
DEGREES = [0, 1, 2, 3, 4, 5]
FORMAL_CHARGES = [-1, 0, 1]
HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
]
NUM_HS = [0, 1, 2, 3, 4]
BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]

_MORGAN_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def _parse(smiles: object) -> Chem.Mol | None:
    """Parse SMILES into RDKit Mol object, returning None if unparsable"""
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        return None
    return mol


def canonicalize(smiles: str) -> str | None:
    """Return RDKit's canonical SMILES, or None if input is unparsable"""
    mol = _parse(smiles)
    return Chem.MolToSmiles(mol) if mol is not None else None


def _one_hot(value, choices: list) -> list[float]:
    """One-hot encode value over choices plus a final other slot"""
    vec = [0.0] * (len(choices) + 1)
    idx = choices.index(value) if value in choices else len(choices)
    vec[idx] = 1.0
    return vec


def atom_features(atom: Chem.Atom) -> list[float]:
    """Encode one atom as a fixed-length float vector (length ATOM_FEATURE_DIM).

    Layout: element | degree | formal charge | hybridization | total H count
    (each one-hot with a trailing "other" slot), then aromatic and in-ring as
    single 0/1 flags.
    """
    return (
        _one_hot(atom.GetSymbol(), ELEMENTS)
        + _one_hot(atom.GetDegree(), DEGREES)
        + _one_hot(atom.GetFormalCharge(), FORMAL_CHARGES)
        + _one_hot(atom.GetHybridization(), HYBRIDIZATIONS)
        + _one_hot(atom.GetTotalNumHs(), NUM_HS)
        + [float(atom.GetIsAromatic()), float(atom.IsInRing())]
    )


def bond_features(bond: Chem.Bond) -> list[float]:
    """Encode one bond as a fixed-length float vector (length BOND_FEATURE_DIM).

    Layout: bond type one-hot (single/double/triple/aromatic + "other"), then
    conjugated and in-ring as single 0/1 flags.
    """
    return _one_hot(bond.GetBondType(), BOND_TYPES) + [
        float(bond.GetIsConjugated()),
        float(bond.IsInRing()),
    ]


# Feature widths, derived from the choice lists so they can never drift.
# +1 per one-hot for the "other" slot; +2 for the two boolean flags.
ATOM_FEATURE_DIM = (
    len(ELEMENTS)
    + len(DEGREES)
    + len(FORMAL_CHARGES)
    + len(HYBRIDIZATIONS)
    + len(NUM_HS)
    + 5
    + 2
)
BOND_FEATURE_DIM = len(BOND_TYPES) + 1 + 2


def smiles_to_graph(smiles: str) -> Data | None:
    """Convert a SMILES string into a PyG ``Data`` graph, or None if unparsable.

    - ``x``: [num_atoms, ATOM_FEATURE_DIM] float node features
    - ``edge_index``: [2, 2 * num_bonds] long; each bond appears in both
      directions (i->j and j->i) because PyG treats edges as directed
    - ``edge_attr``: [2 * num_bonds, BOND_FEATURE_DIM] float, row-aligned with
      ``edge_index``
    - ``smiles``: the canonical SMILES, kept for traceability

    Hydrogens stay implicit (folded into the total-H atom feature).
    """
    mol = _parse(smiles)
    if mol is None:
        return None

    x = torch.tensor([atom_features(a) for a in mol.GetAtoms()], dtype=torch.float)

    sources: list[int] = []
    targets: list[int] = []
    edge_feats: list[list[float]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = bond_features(bond)
        sources += [i, j]
        targets += [j, i]
        edge_feats += [feats, feats]

    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    # Explicit shape so single-atom molecules (no bonds) still get [0, BOND_FEATURE_DIM].
    edge_attr = torch.tensor(edge_feats, dtype=torch.float).reshape(
        -1, BOND_FEATURE_DIM
    )

    return Data(
        x=x, edge_index=edge_index, edge_attr=edge_attr, smiles=Chem.MolToSmiles(mol)
    )


def smiles_to_fingerprint(smiles: str) -> np.ndarray | None:
    """Return a 2048-bit Morgan fingerprint (radius 2) as a uint8 array, or None."""
    mol = _parse(smiles)
    if mol is None:
        return None
    return _MORGAN_GEN.GetFingerprintAsNumPy(mol).astype(np.uint8)
