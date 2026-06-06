import os.path

import pandas as pd

from sklearn.model_selection import train_test_split


def split_s0(base_dir, itc_file_name, tr=0.7, va=0.1, te=0.2, seed=42):
    itc = pd.read_csv(str(os.path.join(base_dir, itc_file_name)))
    itc = itc.drop_duplicates(itc.columns, keep="first")

    train_df, temp_df = train_test_split(
        itc, test_size=1 - tr, random_state=seed, stratify=itc["label"]
    )

    train_drug = collect_drugs(train_df["drug1"], train_df["drug2"])
    train_dict = {key: i for i, key in enumerate(train_drug)}
    train_df["drug1"] = train_df["drug1"].map(train_dict)
    train_df["drug2"] = train_df["drug2"].map(train_dict)

    val_df, test_df = train_test_split(
        temp_df, test_size=te / (va + te), random_state=seed, stratify=temp_df["label"]
    )

    val_drug = collect_drugs(val_df["drug1"], val_df["drug2"])
    val_dict = {key: i for i, key in enumerate(val_drug)}
    val_df["drug1"] = val_df["drug1"].map(val_dict)
    val_df["drug2"] = val_df["drug2"].map(val_dict)

    test_drug = collect_drugs(test_df["drug1"], test_df["drug2"])
    test_dict = {key: i for i, key in enumerate(test_drug)}
    test_df["drug1"] = test_df["drug1"].map(test_dict)
    test_df["drug2"] = test_df["drug2"].map(test_dict)

    save_splits(
        os.path.join(base_dir, "s0"),
        train_drug,
        val_drug,
        test_drug,
        train_df,
        val_df,
        test_df,
    )


def collect_drugs(*args):
    return pd.concat([*args]).drop_duplicates().reset_index(drop=True)


def save_splits(dir_path, train_drugs, val_drug, test_drug, train, val, test):

    header = ["id"]

    os.makedirs(dir_path, exist_ok=True)

    train_drugs.to_csv(
        os.path.join(dir_path, "train_set.csv"), index=False, header=header
    )
    val_drug.to_csv(os.path.join(dir_path, "val_set.csv"), index=False, header=header)

    test_drug.to_csv(os.path.join(dir_path, "test_set.csv"), index=False, header=header)

    train.to_csv(os.path.join(dir_path, "train.csv"), index=False)
    val.to_csv(os.path.join(dir_path, "val.csv"), index=False)
    test.to_csv(os.path.join(dir_path, "test.csv"), index=False)


if __name__ == "__main__":
    split_s0(
        base_dir="./data",
        itc_file_name="KnownDDI.csv",
        tr=0.7,
        va=0.1,
        te=0.2,
    )
