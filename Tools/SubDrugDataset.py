import pandas as pd

from .DrugDataset import DrugDataset
from torch.utils.data import Dataset
from torch_geometric.data import Batch


class SubDrugDataset(Dataset):
    def __init__(self, sub_drug_path, drug_dataset: DrugDataset):
        self.sub_drug_path = sub_drug_path
        self.indices = pd.read_csv(sub_drug_path)
        self.drug_dataset = drug_dataset

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        return self.drug_dataset[self.indices.loc[idx]["id"]]

    def drug_collate_fn(self, batch):
        return Batch.from_data_list(batch)
