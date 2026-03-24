from functools import lru_cache

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdFingerprintGenerator, rdMolDescriptors

from .settings import FP_SIZE

FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=FP_SIZE)

SMARTS_PATTERNS: dict[str, Chem.Mol] = {
    "carboxylic_acid": Chem.MolFromSmarts("[CX3](=O)[OX2H]"),
    "pyridine_like": Chem.MolFromSmarts("n1ccccc1"),
    "primary_secondary_amine": Chem.MolFromSmarts("[NX3;H2,H1]"),
    "phenol": Chem.MolFromSmarts("c[OH]"),
    "amide": Chem.MolFromSmarts("C(=O)N"),
}

FLAG_NAMES = list(SMARTS_PATTERNS)
ACID_IDX = FLAG_NAMES.index("carboxylic_acid")
PYRIDINE_IDX = FLAG_NAMES.index("pyridine_like")
AMINE_IDX = FLAG_NAMES.index("primary_secondary_amine")

BASIC_DESC_NAMES = [
    "mw",
    "logp",
    "tpsa",
    "hbd",
    "hba",
    "rot",
    "fracsp3",
    "narom",
    "nheavy",
    "formal_charge",
    "n_n",
    "n_o",
]
GASTEIGER_NAMES = ["gasteiger_max", "gasteiger_min", "gasteiger_pos_sum", "gasteiger_neg_sum"]
ADVANCED_NAMES = [
    "num_f",
    "num_cl",
    "num_br",
    "polar_surface_fraction",
    "complexity",
    "is_perfluoro",
    "labute_asa",
]
PAIR_BINARY_NAMES = ["acid_base_any", "acid_acid", "base_base"]
INTERACTION_NAMES = [
    "halogen_bond_proxy",
    "size_mismatch",
    "hbond_cross_sum",
    "hbond_cross_max",
    "tanimoto",
]


def canonicalize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, canonical=True)


def parse_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return mol


def basic_descriptors(mol: Chem.Mol) -> np.ndarray:
    return np.array(
        [
            Descriptors.MolWt(mol),
            Descriptors.MolLogP(mol),
            rdMolDescriptors.CalcTPSA(mol),
            rdMolDescriptors.CalcNumHBD(mol),
            rdMolDescriptors.CalcNumHBA(mol),
            rdMolDescriptors.CalcNumRotatableBonds(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
            rdMolDescriptors.CalcNumAromaticRings(mol),
            mol.GetNumHeavyAtoms(),
            Chem.GetFormalCharge(mol),
            sum(atom.GetAtomicNum() == 7 for atom in mol.GetAtoms()),
            sum(atom.GetAtomicNum() == 8 for atom in mol.GetAtoms()),
        ],
        dtype=float,
    )


def molecule_flags(mol: Chem.Mol) -> np.ndarray:
    return np.array([mol.HasSubstructMatch(pattern) for pattern in SMARTS_PATTERNS.values()], dtype=float)


def gasteiger_stats(mol: Chem.Mol) -> np.ndarray:
    charged = Chem.Mol(mol)
    AllChem.ComputeGasteigerCharges(charged)
    values: list[float] = []
    for atom in charged.GetAtoms():
        try:
            value = float(atom.GetProp("_GasteigerCharge"))
        except Exception:
            value = 0.0
        if not np.isfinite(value):
            value = 0.0
        values.append(value)

    charges = np.array(values, dtype=float)
    return np.array(
        [
            charges.max(initial=0.0),
            charges.min(initial=0.0),
            charges[charges > 0].sum(),
            charges[charges < 0].sum(),
        ],
        dtype=float,
    )


def advanced_descriptors(mol: Chem.Mol) -> np.ndarray:
    num_f = sum(atom.GetAtomicNum() == 9 for atom in mol.GetAtoms())
    num_cl = sum(atom.GetAtomicNum() == 17 for atom in mol.GetAtoms())
    num_br = sum(atom.GetAtomicNum() == 35 for atom in mol.GetAtoms())
    labute_asa = rdMolDescriptors.CalcLabuteASA(mol)
    tpsa = rdMolDescriptors.CalcTPSA(mol)
    return np.array(
        [
            num_f,
            num_cl,
            num_br,
            tpsa / labute_asa if labute_asa > 0 else 0.0,
            Descriptors.BertzCT(mol),
            float(num_f > 4),
            labute_asa,
        ],
        dtype=float,
    )


@lru_cache(maxsize=100_000)
def molecule_features(smiles: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normalized = canonicalize_smiles(smiles)
    mol = parse_mol(normalized)
    fingerprint = np.array(FP_GEN.GetCountFingerprintAsNumPy(mol) > 0, dtype=np.int8)
    return (
        basic_descriptors(mol),
        molecule_flags(mol),
        gasteiger_stats(mol),
        advanced_descriptors(mol),
        fingerprint,
    )


def feature_names() -> list[str]:
    names: list[str] = []
    names.extend(f"fp_diff_{idx}" for idx in range(FP_SIZE))
    names.extend(f"fp_sum_{idx}" for idx in range(FP_SIZE))
    names.extend(f"fp_inter_{idx}" for idx in range(FP_SIZE))
    names.extend(f"desc_sum_{name}" for name in BASIC_DESC_NAMES)
    names.extend(f"desc_diff_{name}" for name in BASIC_DESC_NAMES)
    names.extend(f"desc_prod_{name}" for name in BASIC_DESC_NAMES)
    names.extend(f"flags_or_{name}" for name in FLAG_NAMES)
    names.extend(f"flags_and_{name}" for name in FLAG_NAMES)
    names.extend(f"flags_xor_{name}" for name in FLAG_NAMES)
    names.extend(PAIR_BINARY_NAMES)
    names.extend(f"g_sum_{name}" for name in GASTEIGER_NAMES)
    names.extend(f"g_diff_{name}" for name in GASTEIGER_NAMES)
    names.extend(f"adv_sum_{name}" for name in ADVANCED_NAMES)
    names.extend(f"adv_diff_{name}" for name in ADVANCED_NAMES)
    names.extend(INTERACTION_NAMES)
    return names


def pair_features(smiles1: str, smiles2: str) -> np.ndarray:
    desc1, flags1, g1, adv1, fp1 = molecule_features(smiles1)
    desc2, flags2, g2, adv2, fp2 = molecule_features(smiles2)

    acid_base_any = float(
        (flags1[ACID_IDX] and (flags2[PYRIDINE_IDX] or flags2[AMINE_IDX]))
        or (flags2[ACID_IDX] and (flags1[PYRIDINE_IDX] or flags1[AMINE_IDX]))
    )
    acid_acid = float(flags1[ACID_IDX] and flags2[ACID_IDX])
    base_base = float(
        (flags1[PYRIDINE_IDX] or flags1[AMINE_IDX]) and (flags2[PYRIDINE_IDX] or flags2[AMINE_IDX])
    )

    halogens1 = adv1[0] + adv1[1] + adv1[2]
    halogens2 = adv2[0] + adv2[1] + adv2[2]
    nitrogens1 = desc1[10]
    nitrogens2 = desc2[10]

    mol1 = parse_mol(canonicalize_smiles(smiles1))
    mol2 = parse_mol(canonicalize_smiles(smiles2))

    dense_features = np.concatenate(
        [
            desc1 + desc2,
            np.abs(desc1 - desc2),
            desc1 * desc2,
            np.maximum(flags1, flags2),
            flags1 * flags2,
            np.abs(flags1 - flags2),
            np.array([acid_base_any, acid_acid, base_base], dtype=float),
            g1 + g2,
            np.abs(g1 - g2),
            adv1 + adv2,
            np.abs(adv1 - adv2),
            np.array(
                [
                    halogens1 * nitrogens2 + halogens2 * nitrogens1,
                    max(desc1[0], desc2[0]) / (min(desc1[0], desc2[0]) + 1e-6),
                    desc1[3] * desc2[4] + desc2[3] * desc1[4],
                    max(desc1[3] * desc2[4], desc2[3] * desc1[4]),
                    DataStructs.TanimotoSimilarity(FP_GEN.GetFingerprint(mol1), FP_GEN.GetFingerprint(mol2)),
                ],
                dtype=float,
            ),
        ]
    )

    return np.concatenate(
        [
            np.abs(fp1 - fp2).astype(float),
            (fp1 + fp2).astype(float),
            (fp1 * fp2).astype(float),
            dense_features,
        ]
    )


def prepare_features(frame: pd.DataFrame) -> pd.DataFrame:
    missing = {"SMILES1", "SMILES2"} - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[np.ndarray] = []
    errors: list[str] = []
    for idx, row in frame.iterrows():
        try:
            rows.append(pair_features(str(row["SMILES1"]), str(row["SMILES2"])))
        except Exception as exc:
            errors.append(f"row={idx}: {exc}")

    if errors:
        preview = "\n".join(errors[:5])
        raise ValueError(f"Feature generation failed.\n{preview}\nTotal bad rows: {len(errors)}")

    return pd.DataFrame(np.vstack(rows), columns=feature_names(), index=frame.index)


def make_groups(frame: pd.DataFrame) -> np.ndarray:
    groups: list[str] = []
    for _, row in frame.iterrows():
        left = canonicalize_smiles(str(row["SMILES1"]))
        right = canonicalize_smiles(str(row["SMILES2"]))
        groups.append("||".join(sorted((left, right))))
    return np.array(groups)
