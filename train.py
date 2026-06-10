import os
import random
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.nn import CrossEntropyLoss
import numpy as np
import torch
from torch.utils.data import DataLoader
from process_data import Timer, load_data, drug_collate_fn, itc_collate_fn
import model
from model import EarlyStop, Classifier
from typing import List
import config
from config import Config
import logging

logger = logging.getLogger("train")


def train_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    optimizer,
    criterion,
    device,
    scaler=None,
):

    encoder.train()
    classifier.train()

    train_loss = 0.0
    train_acc = 0.0
    total_batch = len(itc_loader)
    batch_counter = 0
    for d1, d2, labels in itc_loader:
        batch_counter += 1
        d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
        optimizer.zero_grad()
        if scaler is not None:
            with torch.autocast(device_type="cuda"):
                all_drugs = torch.cat(
                    [encoder(drugs.to(device)) for drugs in drug_loader]
                )
                logits = classifier(all_drugs[d1], all_drugs[d2])
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)

        acc = (preds == labels).float().mean()

        train_loss += loss.item()
        train_acc += acc.item()
        print(
            f"\r[Train] [Batch:{batch_counter}/{total_batch}] loss:{loss.item():.5f},acc:{acc.item():.5f}",
            end="",
            flush=True,
        )
    print()
    avg_train_loss = train_loss / len(itc_loader)
    avg_train_acc = train_acc / len(itc_loader)
    logger.info(
        "Epoch train done  |  avg_loss=%.5f  avg_acc=%.5f  (%d batches)",
        avg_train_loss,
        avg_train_acc,
        total_batch,
    )
    return avg_train_loss, avg_train_acc


def val_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    criterion,
    device,
):
    encoder.eval()
    classifier.eval()

    val_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in drug_loader])

        for d1, d2, labels in itc_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            val_loss += loss.item()
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(prob.cpu())
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_labels = torch.cat(all_labels, dim=0).numpy()
    all_probs = torch.cat(all_probs, dim=0).numpy()
    avg_loss = val_loss / len(itc_loader)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    cm = confusion_matrix(all_labels, all_preds)
    logger.info(
        "Validation done  |  loss=%.5f  acc=%.5f  f1=%.5f  auc=%.5f",
        avg_loss,
        acc,
        f1,
        auc,
    )
    return (avg_loss, acc, f1, auc, cm)


def train(
    config_class_name: str,
    config: Config,
    history=None,
):
    name = config_class_name
    encoder_type = config.encoder
    data_source = config.data_source
    split_type = config.split_type
    epochs = config.epochs
    node_dim = config.node_dim
    edge_dim = config.edge_dim
    graph_dim = config.graph_dim
    d_model = config.d_model
    lr = config.lr
    heads = config.heads
    dp_r = config.dp_r
    train_size = config.train_size
    seed = config.seed
    block_num = config.block_num
    class_num = config.class_num
    drug_batch_size = config.drug_batch_size
    itc_batch_size = config.itc_batch_size
    label_smoothing = config.label_smoothing
    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_epoch = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_itc_generator = torch.Generator()
    train_itc_generator.manual_seed(seed)

    pin_memory = True if torch.cuda.is_available() else False

    logger.info(
        "Train config | epochs=%d  encoder=%s  data_source=%s  split_type=%s  device=%s  seed=%d",
        epochs,
        encoder_type,
        data_source,
        split_type,
        device,
        seed,
    )
    base_dir = os.path.join("./checkpoints", name)
    os.makedirs(base_dir, exist_ok=True)
    best_path = os.path.join(base_dir, "best.pt")
    history_path = os.path.join(base_dir, "history.pt")
    result_path = os.path.join(base_dir, "result.csv")
    cm_path = os.path.join(base_dir, "confusion_matrix.csv")

    drug_set, train_itc, val_itc, _, _ = load_data(
        data_source, split_type, train_size, seed
    )
    assert drug_set is not None and train_itc is not None and val_itc is not None
    drug_loader = DataLoader(
        drug_set,
        collate_fn=drug_collate_fn,
        batch_size=drug_batch_size,
        pin_memory=pin_memory,
        num_workers=2,
        shuffle=False,
    )
    train_loader = DataLoader(
        train_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        pin_memory=pin_memory,
        num_workers=2,
        shuffle=True,
        generator=train_itc_generator,
    )

    val_loader = DataLoader(
        val_itc,
        collate_fn=itc_collate_fn,
        batch_size=itc_batch_size,
        pin_memory=pin_memory,
        num_workers=2,
        shuffle=False,
    )
    encoder = getattr(model, encoder_type)
    if encoder_type == "AttnGINTFEncoder":
        encoder = encoder(
            node_dim, edge_dim, graph_dim, d_model, block_num, dp_r, heads
        ).to(device)
    else:
        encoder = encoder(node_dim, d_model, block_num, dp_r).to(device)

    classifier = Classifier(d_model, class_num, dp_r).to(device)
    optimizer = Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, epochs, eta_min=0.00001)
    criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
    early_stop = EarlyStop(patience=10, mode="max", min_delta=0.001)
    scaler = torch.GradScaler() if torch.cuda.is_available() else None

    result = {
        "train_loss": [],
        "train_acc": [],
        "train_timer": [],
        "val_loss": [],
        "val_acc": [],
        "val_f1_score": [],
        "val_auc": [],
        "val_timer": [],
    }

    total_timer = 0

    if history is not None:
        current_checkpoint = history
        start_epoch = current_checkpoint["epoch"]
        encoder.load_state_dict(history["encoder"])
        classifier.load_state_dict(history["classifier"])
        optimizer.load_state_dict(history["optimizer"])
        scheduler.load_state_dict(history["scheduler"])
        early_stop.load_state_dict(history["early_stop"])
        if scaler is not None and history["scaler"] is not None:
            scaler.load_state_dict(history["scaler"])

        if (
            "train_itc_generator" in history
            and history["train_itc_generator"] is not None
        ):
            train_itc_generator.set_state(history["train_itc_generator"])

        if torch.cuda.is_available() and history["cuda_random"] is not None:
            torch.cuda.set_rng_state_all(history["cuda_random"])
        torch.random.set_rng_state(history["torch_random"])
        np.random.set_state(history["numpy_random"])
        random.setstate(history["python_random"])
        logger.info("Resume checkpoint loaded  |  epoch=%d", start_epoch)

    if early_stop.early_stop:
        logger.info(
            "Early stop already triggered at epoch %d/%d — nothing to resume",
            start_epoch,
            epochs,
        )
        return

    if start_epoch >= epochs:
        logger.info(
            "Experiment already finished at epoch %d/%d — nothing to resume",
            start_epoch,
            epochs,
        )
        return

    if history is not None and history["result"] is not None:
        df = pd.read_csv(result_path)
        result = df.to_dict(orient="list")
        total_timer = sum(result["train_timer"]) + sum(result["val_timer"])

    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        with Timer() as timer:
            train_loss, train_acc = train_one_epoch(
                encoder,
                classifier,
                drug_loader,
                train_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
        logger.info(
            "Epoch %d/%d  |  train loss=%.5f  acc=%.5f  (%.2fs)",
            current_epoch,
            epochs,
            train_loss,
            train_acc,
            timer.elapsed,
        )
        total_timer += timer.elapsed
        result["train_loss"].append(train_loss)
        result["train_acc"].append(train_acc)
        result["train_timer"].append(timer.elapsed)

        with Timer() as timer:
            val_loss, val_acc, val_f1_score, val_auc, cm = val_one_epoch(
                encoder,
                classifier,
                drug_loader,
                val_loader,
                criterion,
                device,
            )

        logger.info(
            "Epoch %d/%d  |  val  loss=%.5f  acc=%.5f  f1=%.5f  auc=%.5f  (%.2fs)",
            current_epoch,
            epochs,
            val_loss,
            val_acc,
            val_f1_score,
            val_auc,
            timer.elapsed,
        )
        total_timer += timer.elapsed
        result["val_loss"].append(val_loss)
        result["val_acc"].append(val_acc)
        result["val_f1_score"].append(val_f1_score)
        result["val_auc"].append(val_auc)
        result["val_timer"].append(timer.elapsed)

        scheduler.step()
        is_improved = early_stop(val_f1_score)

        if is_improved:
            torch.save(
                {
                    "epoch": current_epoch,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                best_path,
            )
            cm_df = pd.DataFrame(
                cm,
                index=[f"True_{i}" for i in range(class_num)],
                columns=[f"Pred_{i}" for i in range(class_num)],
            )
            cm_df.to_csv(cm_path)
            logger.info(
                "Epoch %d/%d  |  best model improved → saved best.pt",
                current_epoch,
                epochs,
            )
        else:
            logger.info(
                "Epoch %d/%d  |  no improvement  (%d/%d patience)",
                current_epoch,
                epochs,
                early_stop.counter,
                early_stop.patience,
            )

        checkpoint = {
            "epoch": current_epoch,
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "early_stop": early_stop.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "cuda_random": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
            "torch_random": torch.random.get_rng_state(),
            "numpy_random": np.random.get_state(),
            "python_random": random.getstate(),
            "train_itc_generator": train_itc_generator.get_state(),
        }

        if current_epoch % 5 == 0:
            torch.save(checkpoint, history_path)
            pd.DataFrame(result).to_csv(result_path, index=False)
            logger.info(
                "Epoch %d/%d  |  checkpoint saved (history.pt + result.csv)",
                current_epoch,
                epochs,
            )

        logger.info(
            "Epoch %d/%d  |  cumulative time: %.2fs",
            current_epoch,
            epochs,
            total_timer,
        )

        if early_stop.early_stop:
            logger.info(
                "Epoch %d/%d  |  early stop triggered",
                current_epoch,
                epochs,
            )
            break


def resume_training(config_class_name: str):
    cfg: Config
    try:
        cfg = getattr(config, config_class_name)
    except AttributeError:
        logger.warning("Config '%s' not found in config.py", config_class_name)
        return
    history_path = os.path.join("./checkpoints", config_class_name, "history.pt")
    result_path = os.path.join("./checkpoints", config_class_name, "history.pt")
    if not os.path.exists(history_path) or not os.path.exists(result_path):
        logger.warning(
            "history.pt or result.csv not found for config %s,resume_training failed ",
            config_class_name,
        )
        return
    history = torch.load(history_path, weights_only=False)
    result = pd.read_csv(result_path)
    history["result"] = result
    train(config_class_name, cfg, history=history)


def run_training(config_class_name: str):
    cfg: Config
    try:
        cfg = getattr(config, config_class_name)
    except AttributeError:
        logger.warning("Config '%s' not found in config.py", config_class_name)
        return
    train(config_class_name, cfg)


def train_all(config_class_name_arr: List[str]):
    for name in config_class_name_arr:
        run_training(name)


__all__ = ["resume_training", "run_training"]
