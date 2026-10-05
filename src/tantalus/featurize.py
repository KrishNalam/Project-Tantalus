"""SMILES --> canonical SMILES, molecular graphs and Morgan fingerprints"""

import numpy as np  # fingerprint returned as numpy array
import torch  #
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from torch_geometric.data import Data

ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P"]
