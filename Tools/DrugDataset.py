import torch
from rdkit import Chem
from rdkit.Chem import  Descriptors
from torch_geometric.data import Data, Batch
import pandas as pd
from torch.utils.data import Dataset
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.warning')

class DrugDataset(Dataset):
    def __init__(self, node_path, add_global_features=False):
        self.node_path = node_path
        self.drug = pd.read_csv(node_path)
        self.add_global_features = add_global_features
        # self.pt = Chem.GetPeriodicTable()
        self.drug["data"] = [self.sdf_to_graph(drug_id) for drug_id in self.drug["id"]]

    def __len__(self):
        return len(self.drug)

    def __getitem__(self, idx):
        if isinstance(idx, str) and idx.startswith("DB"):
            row = self.drug[self.drug['id'] == idx]
            if len(row) == 0:
                raise KeyError(f"ID {idx} not found")
            return row.iloc[0]['data']
        return self.drug.loc[idx]["data"]

    def one_hot_encoding(self, x, allowable_set):
        if x not in allowable_set:
            x = allowable_set[-1]
        return [int(x == s) for s in allowable_set]

    def atom_features(self, atom):
        features = []

        # 1. 原子类型 (10)
        features += self.one_hot_encoding(
            atom.GetSymbol(), ["C", "N", "O", "S", "F", "P", "Cl", "Br", "I", "Other"]
        )

        # 2. 度 (6)
        features += self.one_hot_encoding(
            atom.GetDegree(),
            [0, 1, 2, 3, 4, 5, 6],
        )

        # 3. 总氢原子数 (5)
        features += self.one_hot_encoding(
            atom.GetTotalNumHs(),
            [0, 1, 2, 3, 4, 5],
        )

        # 4. 形式电荷 (5)
        features += self.one_hot_encoding(
            atom.GetFormalCharge(),
            [-2, -1, 0, 1, 2],
        )

        # 5. 芳香性 (1)
        features.append(int(atom.GetIsAromatic()))

        # 6. 是否在环中 (1)
        features.append(int(atom.IsInRing()))

        # 7. 杂化类型 (5)
        features += self.one_hot_encoding(
            atom.GetHybridization(),
            [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.OTHER,
            ],
        )

        # 8. 手性中心 (3)
        features += self.one_hot_encoding(
            atom.GetChiralTag(),
            [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
            ],
        )

        # 9.原子质量 (1)
        features.append(atom.GetMass() / 100.0)

        return features

    def bond_features(self, bond):
        bond_type = bond.GetBondType()

        features = [
            bond_type == Chem.rdchem.BondType.SINGLE,
            bond_type == Chem.rdchem.BondType.DOUBLE,
            bond_type == Chem.rdchem.BondType.TRIPLE,
            bond_type == Chem.rdchem.BondType.AROMATIC,
            bond.GetIsConjugated(),
            bond.IsInRing(),
        ]

        # 键立体化学
        features += self.one_hot_encoding(
            bond.GetStereo(),
            [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
                Chem.rdchem.BondStereo.STEREOCIS,
            ],
        )

        return features

    def sdf_to_graph(self, drug_id):
        sdf_path = f"./data/drug_sdf/{drug_id}.sdf"
        supplier = Chem.SDMolSupplier(sdf_path, sanitize=False)
        mol = next(supplier)
        if mol is None:
            print(f"读取失败:{sdf_path}")
            return None

        try:
            mol.UpdatePropertyCache(strict=False)
        except:
            print(f"UpdatePropertyCache失败:{sdf_path}")
            return None


        # 节点特征
        x = []
        for atom in mol.GetAtoms():
            x.append(self.atom_features(atom))
        x = torch.tensor(x, dtype=torch.float)

        # 边特征
        edge_index = []
        edge_attr = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            bf = self.bond_features(bond)

            edge_index.append([i, j])
            edge_index.append([j, i])
            edge_attr.append(bf)
            edge_attr.append(bf)

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        # 构建Data对象
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        # 添加全局分子特征
        if self.add_global_features:
            try:
                global_features = torch.tensor(
                    [
                        Descriptors.MolWt(mol) / 500.0,
                        Descriptors.MolLogP(mol) / 10.0,
                        Descriptors.TPSA(mol) / 200.0,
                        Descriptors.NumHDonors(mol) / 10.0,
                        Descriptors.NumHAcceptors(mol) / 10.0,
                        Descriptors.NumRotatableBonds(mol) / 20.0,
                        Chem.rdMolDescriptors.CalcNumRings(mol) / 10.0,
                    ],
                    dtype=torch.float,
                ).unsqueeze(0)
                data.global_features = global_features
            except:
                # 全局描述符异常填充0向量
                data.global_features = torch.zeros((1, 7))

        return data

    def drug_collate_fn(self, batch):
        return Batch.from_data_list(batch)
