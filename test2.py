import time

import torch
from process_data import (
    DrugDataset,
    InteractionDataset,
    drug_collate_fn,
    itc_collate_fn,
    smiles_to_graph
)
from torch.utils.data import DataLoader


def test_drug_num_workers(num_workers):

    drug_set = DrugDataset("./split_data/drugbank-random-42")
    drug_loader = DataLoader(
        drug_set,
        batch_size=2048,
        num_workers=num_workers,
        shuffle=False,
        collate_fn=drug_collate_fn,
    )
    start = time.time()
    for drug in drug_loader:
        pass
    print(f"num_workers={num_workers}: {time.time() - start:.2f}s")


def test_itc_num_workers(num_workers):

    start = time.time()
    itc_set = InteractionDataset("./split_data/drugbank-random-42", "train")
    itc_loader = DataLoader(
        itc_set,
        batch_size=20480,
        num_workers=num_workers,
        shuffle=True,
        collate_fn=itc_collate_fn,
    )

    for itc in itc_loader:
        pass
    print(f"num_workers={num_workers}: {time.time() - start:.2f}s")

def has_nan_inf():
    drug_set = DrugDataset("./split_data/drugbank-random-42")
    drug_loader = DataLoader(
        drug_set,
        batch_size=2048,
        num_workers=0,
        shuffle=False,
        collate_fn=drug_collate_fn,
    )
    for drug in drug_loader:
        x_has_nan = torch.isnan(drug.x).any().item()  
        edge_attr_has_nan = torch.isnan(drug.edge_attr).any().item()  
        graph_attr_has_nan = torch.isnan(drug.graph_attr).any().item()  
        x_has_inf = torch.isinf(drug.x).any().item()  
        edge_attr_has_inf = torch.isinf(drug.edge_attr).any().item()  
        graph_attr_has_inf = torch.isinf(drug.graph_attr).any().item() 

        print("x_has_nan",x_has_nan)
        print("edge_attr_has_nan",edge_attr_has_nan)
        print("graph_attr_has_nan",graph_attr_has_nan)
        print("x_has_inf",x_has_inf)
        print("edge_attr_has_inf",edge_attr_has_inf)
        print("graph_attr_has_inf",graph_attr_has_inf)

if __name__ == "__main__":
    mol = smiles_to_graph("CN1CCN(CC(=O)N2C3=CC=CC=C3C(=O)NC3=C2N=CC=C3)CC1"	"CN1CCN2C(C1)C1=CC=CC=C1CC1=CC=CC=C21")
    print(mol)
    # drug_set = DrugDataset("./split_data/drugbank-random-42")
    # print