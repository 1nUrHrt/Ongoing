import os
import pandas as pd
import torch
import config
from process_data import load_data, drug_collate_fn, itc_collate_fn
import model
from model import Classifier
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import logging
from config import Config

logger = logging.getLogger("test")


def test(config_class_name: str, config: Config):
    name = config_class_name
    encoder_type = config.encoder
    data_source = config.data_source
    split_type = config.split_type
    node_dim = config.node_dim
    edge_dim = config.edge_dim
    h_dim = config.h_dim
    heads = config.heads
    dp_r = config.dp_r
    seed = config.seed
    block_num = config.block_num
    class_num = config.class_num
    drug_batch_size = config.drug_batch_size
    itc_batch_size = config.itc_batch_size
    label_smoothing = config.label_smoothing

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(
        "Testing  |  name=%s  encoder=%s  device=%s  data_source=%s split_type=%s",
        name,
        encoder_type,
        device,
        data_source,
        split_type,
    )

    pin_memory = True if torch.cuda.is_available() else False
    _, _, _, drug_set, test_itc = load_data(data_source, split_type, None, seed=seed)
    assert drug_set is not None and test_itc is not None
    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=drug_batch_size,
        pin_memory=pin_memory,
        num_workers=2,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        pin_memory=pin_memory,
        num_workers=2,
        shuffle=False,
    )
    encoder = getattr(model, encoder_type)
    if encoder_type == "AttnEncoder":
        encoder = encoder(node_dim, edge_dim, h_dim, block_num, dp_r, heads).to(device)
    else:
        encoder = encoder(node_dim, h_dim, block_num, dp_r, heads).to(device)
    classifier = Classifier(h_dim, class_num, dp_r).to(device)
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
    base_dir = os.path.join("./checkpoints", name)
    best_path = os.path.join(base_dir, "best.pt")
    evaluate_path = os.path.join(base_dir, "evaluate.csv")
    evaluate = {}
    if not os.path.exists(best_path):
        logger.warning(
            "Best model not found: %s/best.pt — skipping evaluation", base_dir
        )
        return

    best_model = torch.load(best_path, weights_only=False)
    encoder.load_state_dict(best_model["encoder"])
    classifier.load_state_dict(best_model["classifier"])

    encoder.eval()
    classifier.eval()

    logger.info("Evaluating on test set  |  best_epoch=%d", best_model["epoch"])

    test_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in test_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            test_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(prob.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    evaluate["best_epoch"] = best_model["epoch"]
    evaluate["test_loss"] = test_loss / len(test_loader)
    evaluate["test_acc"] = accuracy_score(all_labels, all_preds)
    evaluate["test_f1_score"] = f1_score(
        all_labels, all_preds, average="macro", zero_division=0
    )
    evaluate["test_auc"] = roc_auc_score(
        all_labels, all_probs, multi_class="ovr", average="macro"
    )
    pd.DataFrame(evaluate).to_csv(evaluate_path, index=False)
    logger.info(
        "Test results  |  loss=%.5f  acc=%.5f  f1=%.5f  auc=%.5f  -> %s",
        evaluate["test_loss"],
        evaluate["test_acc"],
        evaluate["test_f1_score"],
        evaluate["test_auc"],
        evaluate_path,
    )


def run_test(config_class_name: str):
    cfg: Config
    try:
        cfg = getattr(config, config_class_name)
    except AttributeError:
        logger.warning("Config '%s' not found in config.py", config_class_name)
        return
    test(config_class_name, cfg)


__all__ = ["run_test"]
