import os

import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def heatmap(file_name: str):

    out_dir = os.path.join("./graph", file_name)
    os.makedirs(out_dir, exist_ok=True)
    cm_df = pd.read_csv(f"./checkpoints/{file_name}/confusion_matrix.csv", index_col=0)

    cm = cm_df.values

    cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    # 绘图
    plt.figure(figsize=(20, 18))
    plt.imshow(cm, cmap="Blues", aspect="auto", interpolation="nearest")
    plt.colorbar(label="Count")
    plt.title("Confusion Matrix (86 classes)")
    plt.xlabel("Predicted")
    plt.ylabel("True")

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "cm.png"), dpi=150)


def line_chart(file_name: str):
    out_dir = os.path.join("./graph", file_name)
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(f"./checkpoints/{file_name}/result.csv")
    print(df.columns)

    fig_type = {
        "Loss": ["train_loss", "val_loss"],
        "Acc": ["train_acc", "val_acc"],
        "F1_Score": ["val_f1_score"],
        "Auc": ["val_auc"],
    }
    for type, cols in fig_type.items():
        type_df = df[cols]
        for col in type_df.columns:
            plt.plot(type_df.index + 1, type_df[col], label=col.replace("_", " "))
        plt.xlabel("Epoch")
        plt.ylabel(type.replace("_", " "))
        plt.title(f"训练 & 验证{type}变化曲线")
        plt.legend()
        plt.savefig(
            os.path.join(out_dir, f"{type}.png"),
            dpi=300,
            bbox_inches="tight",
        )
        plt.clf()


if __name__ == "__main__":
    file_names = ["attn_gin_tf"]
    for i in file_names:
        line_chart(i)
        heatmap(i)
