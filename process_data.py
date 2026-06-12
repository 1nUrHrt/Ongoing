import math
import os.path
from typing import Literal
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, Lipinski, GraphDescriptors,ValenceType
from torch_geometric.data import Data, InMemoryDataset, Batch
from rdkit.Chem import rdCIPLabeler
import time


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter()
        self.elapsed = self.end - self.start


class DrugDataset(InMemoryDataset):
    def __init__(
        self,
        root,
        type: Literal["train", "val", "test"] = "train",
        transform=None,
        pre_transform=None,
    ):
        self.file_name = f"{type}_drug.csv"
        self.proc_name = f"{type}_drug.pt"
        super().__init__(root, transform, pre_transform)
        self._data, self.slices = torch.load(
            self.processed_paths[0], weights_only=False
        )

    @property
    def processed_file_names(self):
        return [self.proc_name]

    @property
    def raw_file_names(self):
        return [self.file_name]

    def download(self):
        pass

    def process(self):
        df = pd.read_csv(self.raw_paths[0])
        data_list = []
        for smile in df["smile"]:
            mol = smiles_to_graph(smile)
            data_list.append(mol)
        self._data, self.slices = self.collate(data_list)
        torch.save((self._data, self.slices), self.processed_paths[0])


class InteractionDataset(Dataset):
    def __init__(self, root, type: Literal["train", "val", "test"] = "train"):
        super().__init__()
        cache_key = f"{type}_itc.pt"
        cache_file_path = os.path.join(root, "processed", cache_key)
        if not os.path.exists(cache_file_path):
            os.makedirs(os.path.join(root, "processed"), exist_ok=True)
            df = pd.read_csv(os.path.join(root, "raw", f"{type}_itc.csv"))
            drug1 = torch.tensor(df["drug1"].values, dtype=torch.long)
            drug2 = torch.tensor(df["drug2"].values, dtype=torch.long)
            label = torch.tensor(df["label"].values, dtype=torch.long)
            torch.save((drug1, drug2, label), cache_file_path)

        drug1, drug2, label = torch.load(cache_file_path, weights_only=False)
        self.drug1 = drug1.share_memory_()
        self.drug2 = drug2.share_memory_()
        self.label = label.share_memory_()

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        return self.drug1[idx], self.drug2[idx], self.label[idx]


def _load_data(root: str, type: Literal["train", "val", "test"] = "train"):
    return DrugDataset(root, type), InteractionDataset(root, type)


def load_data(
    data_source: Literal["drugbank", "twosides"],
    split_type: Literal["random", "cluster"],
    type: Literal["train", "val", "test"] = "train",
    seed=42,
):
    base_dir = os.path.join(
        "./split_data", data_source + "-" + split_type + "-" + str(seed)
    )
    return _load_data(base_dir, type)


def split_data(
    data_source: Literal["drugbank", "twosides"] = "drugbank",
    split_type: Literal["random", "cluster"] = "random",
    train_size=0.8,
    seed=42,
):
    save_dir = os.path.join(
        "./split_data", data_source + "-" + split_type + "-" + str(seed), "raw"
    )
    os.makedirs(save_dir, exist_ok=True)
    if data_source == "drugbank":
        if split_type == "random":
            _split_drugbank_random(
                pd.read_csv("./data/drugbank.tab", sep="\t"), train_size, seed, save_dir
            )
        else:
            raise TypeError()
    else:
        raise TypeError()


def _split_drugbank_random(df: pd.DataFrame, train_size, seed, save_dir):
    drug1 = df[["ID1", "X1"]].drop_duplicates(keep="first")
    drug2 = df[["ID2", "X2"]].drop_duplicates(keep="first")
    drug1.columns = ["id", "smile"]
    drug2.columns = ["id", "smile"]
    drug = (
        pd.concat([drug1, drug2])
        .drop_duplicates(subset=["id", "smile"], keep="first")
        .reset_index(drop=True)
    )
    id_map = {row["id"]: row["smile"] for _, row in drug.iterrows()}
    itc = df[["ID1", "ID2", "Y"]].drop_duplicates(keep="first").reset_index(drop=True)
    itc["Y"] = itc["Y"] - 1
    train, test = train_test_split(
        itc, train_size=train_size, random_state=seed, stratify=itc["Y"]
    )

    train_drug = (
        pd.concat([train["ID1"], train["ID2"]], axis=0)
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )
    train_map = {key: i for i, key in enumerate(train_drug)}
    train["ID1"] = train["ID1"].map(train_map)
    train["ID2"] = train["ID2"].map(train_map)
    train_drug = train_drug.map(id_map)

    test_drug = (
        pd.concat([test["ID1"], test["ID2"]], axis=0)
        .drop_duplicates(keep="first")
        .reset_index(drop=True)
    )
    test_map = {key: i for i, key in enumerate(test_drug)}
    test["ID1"] = test["ID1"].map(test_map)
    test["ID2"] = test["ID2"].map(test_map)
    test_drug = test_drug.map(id_map)
    os.makedirs(save_dir, exist_ok=True)
    train_drug.name = "smile"
    train_drug.to_csv(
        os.path.join(save_dir, "train_drug.csv"),
        index=False,
    )
    train.columns = ["drug1", "drug2", "label"]
    train.to_csv(
        os.path.join(save_dir, "train_itc.csv"),
        index=False,
    )

    test_drug.name = "smile"
    test_drug.to_csv(
        os.path.join(save_dir, "test_drug.csv"),
        index=False,
    )
    test.columns = ["drug1", "drug2", "label"]
    test.to_csv(
        os.path.join(save_dir, "test_itc.csv"),
        index=False,
    )


def _one_hot_encoding(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return [int(x == s) for s in allowable_set]


ELECTRONEG = {
    "H": 2.20, "Li": 0.98, "B": 2.04, "C": 2.55, "N": 3.04, "O": 3.44,
    "F": 3.98, "Na": 0.93, "Mg": 1.31, "Al": 1.61, "Si": 1.90, "P": 2.19,
    "S": 2.58, "Cl": 3.16, "K": 0.82, "Ca": 1.00, "Ti": 1.54, "Cr": 1.66,
    "Fe": 1.83, "Co": 1.88, "Cu": 1.90, "Zn": 1.65, "Ga": 1.81, "As": 2.18,
    "Se": 2.55, "Br": 2.96, "Sr": 0.95, "Tc": 1.90, "Ag": 1.93, "Sb": 2.05,
    "I": 2.66, "La": 1.10, "Gd": 1.20, "Pt": 2.28, "Au": 2.54, "Hg": 2.00,
    "Bi": 2.02, "Ra": 0.90
}

def _atom_features(atom):
    features = []

    # 1. Atom symbol (38)
    features += _one_hot_encoding(
        atom.GetSymbol(),
        [
            "H",
            "Li",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "K",
            "Ca",
            "Ti",
            "Cr",
            "Fe",
            "Co",
            "Cu",
            "Zn",
            "Ga",
            "As",
            "Se",
            "Br",
            "Sr",
            "Tc",
            "Ag",
            "Sb",
            "I",
            "La",
            "Gd",
            "Pt",
            "Au",
            "Hg",
            "Bi",
            "Ra",
        ],
    )

    # 2. Degree (7)
    features += _one_hot_encoding(
        atom.GetDegree(),
        [0, 1, 2, 3, 4, 5, 6],
    )

    # 3. Total hydrogens (5)
    features += _one_hot_encoding(
        atom.GetTotalNumHs(),
        [0, 1, 2, 3, 4, 5],
    )

    # 4. Formal charge (5)
    features += _one_hot_encoding(
        atom.GetFormalCharge(),
        [-2, -1, 0, 1, 2],
    )

    # 5. Aromaticity (1)
    features.append(int(atom.GetIsAromatic()))

    # 6. In ring (1)
    features.append(int(atom.IsInRing()))

    # 7. Hybridization (7)
    hyb_list = [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2,
        Chem.rdchem.HybridizationType.OTHER,
    ]
    features += _one_hot_encoding(atom.GetHybridization(), hyb_list)
    # 8. Chiral tag (3)
    features += _one_hot_encoding(
        atom.GetChiralTag(),
        [
            Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        ],
    )

    # 9. Atomic mass (1)
    features.append(atom.GetMass() / 100.0)
    # 10. 显式价态 8维
    features += _one_hot_encoding(
        atom.GetValence(ValenceType.EXPLICIT), [0, 1, 2, 3, 4, 5, 6, 7]
    )
    # 11. 隐式价态 8维
    features += _one_hot_encoding(
        atom.GetValence(ValenceType.IMPLICIT), [0, 1, 2, 3, 4, 5, 6, 7]
    )
    # 12. 是否杂原子 1维 (C/H=0，其余=1)
    symbol = atom.GetSymbol()
    features.append(0 if symbol in ("C", "H") else 1)
    # 13. 原子电负性 1维 (归一化)
    features.append(ELECTRONEG.get(symbol, 2.5) / 4.0)
    # 14. Gasteiger 部分电荷 1维
    try:
        charge = float(atom.GetProp("_GasteigerCharge"))
    except:
        charge = 0.0
    if math.isnan(charge) or math.isinf(charge):
        charge = 0.0
    features.append(charge)

    features.append(
        int(symbol in ("O", "N", "S") and atom.GetTotalNumHs() > 0)
    )

    features.append(int(symbol in ("O", "N", "S", "F")))

    # 17. CIP 手性标签 (3维: R, S, None)
    try:
        cip = atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "None"
    except:
        cip = "None"
    features += _one_hot_encoding(cip, ["R", "S", "None"])

    return features


def _bond_features(bond):
    bond_type = bond.GetBondType()

    features = [
        bond_type == Chem.rdchem.BondType.SINGLE,
        bond_type == Chem.rdchem.BondType.DOUBLE,
        bond_type == Chem.rdchem.BondType.TRIPLE,
        bond_type == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing(),
    ]

    # Bond stereochemistry
    features += _one_hot_encoding(
        bond.GetStereo(),
        [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
            Chem.rdchem.BondStereo.STEREOCIS,
        ],
    )

    # 1. 浮点键级 1维
    features.append(bond.GetBondTypeAsDouble())
    # 2. 是否芳香键（二次强化）1维
    features.append(int(bond.GetIsAromatic()))
    # 3. 共轭环键 1维
    features.append(int(bond.GetIsConjugated() and bond.IsInRing()))
    # 4. 键在 ≤6 元环内 (1维)
    features.append(int(any(bond.IsInRingSize(s) for s in range(3, 7))))
    # 5. 两端原子形式电荷差 (1维)
    f1 = bond.GetBeginAtom().GetFormalCharge()
    f2 = bond.GetEndAtom().GetFormalCharge()
    features.append(f1 - f2)
    # 6. 键是否连接杂原子 (1维)
    sym1 = bond.GetBeginAtom().GetSymbol()
    sym2 = bond.GetEndAtom().GetSymbol()
    # 7. 桥键特征：两端原子环状态不同 (1维)
    r1 = bond.GetBeginAtom().IsInRing()
    r2 = bond.GetEndAtom().IsInRing()
    features.append(int(r1 != r2))
    features.append(0 if sym1 in ("C", "H") and sym2 in ("C", "H") else 1)

    return features


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    mol.UpdatePropertyCache(strict=False)
    AllChem.ComputeGasteigerCharges(mol)
    # Node features
    x = []
    for atom in mol.GetAtoms():
        x.append(_atom_features(atom))
    x = torch.tensor(x, dtype=torch.float)
    # 分配 CIP 标签（用于手性特征）
    try:
        rdCIPLabeler.AssignCIPLabels(mol)
    except:
        pass 

    # Edge features
    edge_index = []
    edge_attr = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bf = _bond_features(bond)

        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_attr.append(bf)
        edge_attr.append(bf)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_attr, dtype=torch.float)

    # ========== 全局特征 ==========
    gw = Descriptors.MolWt(mol) / 500.0
    logp = Descriptors.MolLogP(mol) / 10.0
    tpsa = Descriptors.TPSA(mol) / 250.0
    hdonor = Descriptors.NumHDonors(mol) / 10.0
    haccept = Descriptors.NumHAcceptors(mol) / 10.0
    rot_bond = Descriptors.NumRotatableBonds(mol) / 20.0
    ring_num = rdMolDescriptors.CalcNumRings(mol) / 10.0

    # 重原子数
    heavy_atom = Descriptors.HeavyAtomCount(mol) / 50.0
    # 芳香环数量
    aromatic_ring = rdMolDescriptors.CalcNumAromaticRings(mol) / 10.0
    # 脂肪环数量
    aliphatic_ring = rdMolDescriptors.CalcNumAliphaticRings(mol) / 10.0
    # 摩尔折射率
    mr = Descriptors.MolMR(mol) / 100.0
    # 分子柔性指数
    total_bonds = mol.GetNumBonds()
    frac_rot = Descriptors.NumRotatableBonds(mol) / max(1, total_bonds)
    # 卤素原子总数
    halogens = (
        sum(1 for a in mol.GetAtoms() if a.GetSymbol() in ("F", "Cl", "Br", "I")) / 10.0
    )
    # 氧原子数、氮原子数
    o_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "O") / 10.0
    n_count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N") / 10.0
    # 分子复杂度 (BertzCT)
    complexity = GraphDescriptors.BertzCT(mol) / 1000.0
    # 不饱和碳原子比例
    unsaturated_c = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[C]=[C]')))
    unsat_c_ratio = unsaturated_c / max(1, Descriptors.HeavyAtomCount(mol))
    # Fsp3 (sp3杂化碳比例)
    fsp3 = Lipinski.FractionCSP3(mol) if total_bonds > 0 else 0.0
    # 正/负电荷数
    pos_charge = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() > 0) / 5.0
    neg_charge = sum(1 for a in mol.GetAtoms() if a.GetFormalCharge() < 0) / 5.0
    # 刚性键比例
    rigid_bonds = total_bonds - Descriptors.NumRotatableBonds(mol)
    rigid_ratio = rigid_bonds / max(1, total_bonds)
    # Kappa 形状指数 (归一化)
    kappa1 = GraphDescriptors.Kappa1(mol) / 20.0
    kappa2 = GraphDescriptors.Kappa2(mol) / 20.0
    kappa3 = GraphDescriptors.Kappa3(mol) / 20.0
    # Chi 分子连接性指数 (归一化)
    chi0v = GraphDescriptors.Chi0v(mol) / 10.0
    chi1v = GraphDescriptors.Chi1v(mol) / 10.0
    chi2v = GraphDescriptors.Chi2v(mol) / 10.0

    gen = Chem.rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = gen.GetFingerprint(mol) 
    tensor_fp = torch.tensor(np.array(fp), dtype=torch.float).unsqueeze(0)
    graph_attr = [
        gw, logp, tpsa, hdonor, haccept, rot_bond, ring_num,
        heavy_atom, aromatic_ring, aliphatic_ring, mr, frac_rot,
        halogens, o_count, n_count,
        complexity, unsat_c_ratio, fsp3, pos_charge, neg_charge,
        rigid_ratio, kappa1, kappa2, kappa3, chi0v, chi1v, chi2v
    ]
    graph_attr = torch.tensor(graph_attr, dtype=torch.float).unsqueeze(0)
    graph_attr = torch.cat([graph_attr,tensor_fp],dim=1)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, graph_attr=graph_attr)


def drug_collate_fn(batch):
    return Batch.from_data_list(batch)


def itc_collate_fn(batch):
    drug1, drug2, label = zip(*batch)
    return torch.stack(drug1), torch.stack(drug2), torch.stack(label)


__all__ = ["Timer", "load_data", "split_data", "itc_collate_fn", "drug_collate_fn"]

if __name__ == "__main__":
    load_data("drugbank", "random", "train", 42)
