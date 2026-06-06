import json
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from Tools import (
    get_encoder,
    get_classifier,
    get_optimizer,
    get_scheduler,
    load_dataset,
    get_criterion,
    get_early_stop,
    Timer,
)


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
            with autocast(device_type="cuda"):
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
    return avg_train_loss, avg_train_acc


def val_one_epoch(
    encoder,
    classifier,
    drug_loader,
    itc_loader,
    criterion,
    metric_average,
    device,
):
    encoder.eval()
    classifier.eval()

    scenario = itc_loader.dataset.scenario
    scenario_label = itc_loader.dataset.scenario_label

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
    avg_val_loss = val_loss / len(itc_loader)

    res_dict = {}
    res_dict["total"] = {
        "loss": avg_val_loss,
        "acc": accuracy_score(all_labels, all_preds),
        "f1_score": f1_score(
            all_labels, all_preds, average=metric_average, zero_division=0
        ),
        "auc": roc_auc_score(
            all_labels, all_probs, multi_class="ovr", average=metric_average,labels=range(all_probs.shape[1])
        ),
    }


    for scenario_name in scenario:
        mask = (scenario_label == scenario_name).to_numpy()
        y_true = all_labels[mask]
        y_pred = all_preds[mask]
        y_prob = all_probs[mask]
        res_dict[scenario_name] = {
            "acc": accuracy_score(y_true, y_pred),
            "f1_score": f1_score(
                y_true, y_pred, average=metric_average, zero_division=0
            ),
            "auc": roc_auc_score(
                y_true, y_prob, multi_class="ovr", average=metric_average,labels=range(all_probs.shape[1])
            ),
        }
    return res_dict


def train(config):

    experiment_name = config["name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    epochs = config["epochs"]

    start_epoch = 0

    seed = config["manual_seed"]

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    train_itc_generator = torch.Generator()
    train_itc_generator.manual_seed(seed)

    metric = config["metric"]
    metric_average = config["metric_average"]

    pin_memory = True if torch.cuda.is_available() else False

    print(
        f"[Train Config] Total Epochs:{epochs} Device:{device} Manual Seed:{seed} Metric:{metric} Metric Average:{metric_average}"
    )

    base_dir = os.path.join(config["save_dir"], experiment_name)
    os.makedirs(base_dir, exist_ok=True)
    best_save_path = os.path.join(base_dir, config["best_save_name"])
    current_save_path = os.path.join(base_dir, config["current_save_name"])
    result_dict_path = os.path.join(base_dir, config["result_dict_name"])

    train_set, train_itc, val_set, val_itc, _, _ = load_dataset(**config["dataset"])
    loader_config = config["data_loader"]
    train_set_loader = DataLoader(
        train_set,
        collate_fn=train_set.drug_collate_fn,
        pin_memory=pin_memory,
        **loader_config["train_set"],
    )
    val_set_loader = DataLoader(
        val_set,
        collate_fn=val_set.drug_collate_fn,
        pin_memory=pin_memory,
        **loader_config["val_set"],
    )
    train_itc_loader = DataLoader(
        train_itc,
        collate_fn=train_itc.itc_collate_fn,
        pin_memory=pin_memory,
        generator=train_itc_generator,
        **loader_config["train_itc"],
    )
    val_itc_loader = DataLoader(
        val_itc,
        collate_fn=val_itc.itc_collate_fn,
        pin_memory=pin_memory,
        **loader_config["val_itc"],
    )
    encoder = get_encoder(config["encoder"]).to(device)
    classifier = get_classifier(config["classifier"]).to(device)
    optimizer = get_optimizer(
        config["optimizer"], list(encoder.parameters()) + list(classifier.parameters())
    )
    scheduler = get_scheduler(config["scheduler"], optimizer)
    criterion = get_criterion(config["criterion"])
    early_stop = get_early_stop(config["early_stop"])
    scaler = GradScaler() if torch.cuda.is_available() else None

    result_dict = {
        "train_loss": [],
        "train_acc": [],
        "train_timer": [],
        "val_timer": [],
    }

    total_timer = 0

    if os.path.exists(current_save_path):
        current_checkpoint = torch.load(current_save_path, weights_only=False)
        start_epoch = current_checkpoint["epoch"]
        encoder.load_state_dict(current_checkpoint["encoder"])
        classifier.load_state_dict(current_checkpoint["classifier"])
        optimizer.load_state_dict(current_checkpoint["optimizer"])
        scheduler.load_state_dict(current_checkpoint["scheduler"])
        early_stop.load_state_dict(current_checkpoint["early_stop"])
        if scaler is not None and current_checkpoint["scaler"] is not None:
            scaler.load_state_dict(current_checkpoint["scaler"])

        if (
            "train_itc_generator" in current_checkpoint
            and current_checkpoint["train_itc_generator"] is not None
        ):
            train_itc_generator.set_state(current_checkpoint["train_itc_generator"])

        if torch.cuda.is_available() and current_checkpoint["cuda_random"] is not None:
            torch.cuda.set_rng_state_all(current_checkpoint["cuda_random"])
        torch.random.set_rng_state(current_checkpoint["torch_random"])
        np.random.set_state(current_checkpoint["numpy_random"])
        random.setstate(current_checkpoint["python_random"])
        print("[Load Model] successfully")

    if early_stop.early_stop:
        print(f"[Early Stop] [Epoch:{start_epoch}/{epochs}] trigger early stop")
        return

    if start_epoch >= epochs:
        print(
            f"[Stop Experiment] [Epoch:{start_epoch}/{epochs}] experiment has finished"
        )
        return

    if os.path.exists(result_dict_path):
        df = pd.read_csv(result_dict_path)
        result_dict = df.to_dict(orient="list")
        total_timer = sum(result_dict["train_timer"]) + sum(result_dict["val_timer"])

    print("[Train Srart]")
    for epoch in range(start_epoch, epochs):
        current_epoch = epoch + 1
        print(f"[Current Epoch:{current_epoch}/{epochs}]")
        with Timer() as timer:
            avg_train_loss, avg_train_acc = train_one_epoch(
                encoder,
                classifier,
                train_set_loader,
                train_itc_loader,
                optimizer,
                criterion,
                device,
                scaler,
            )
        print(
            f"[Train] train_loss:{avg_train_loss:.5f},train_acc:{avg_train_acc:.5f},elapsed:{timer.elapsed:.5f} s"
        )
        total_timer += timer.elapsed
        result_dict["train_loss"].append(avg_train_loss)
        result_dict["train_acc"].append(avg_train_acc)
        result_dict["train_timer"].append(timer.elapsed)
        with Timer() as timer:
            res_dict = val_one_epoch(
                encoder,
                classifier,
                val_set_loader,
                val_itc_loader,
                criterion,
                metric_average,
                device,
            )
        total_dict = res_dict["total"]
        S0_dict = res_dict["S0"]
        S1_dict = res_dict["S1"]
        S2_dict = res_dict["S2"]
        print(
            f"[Val Total] loss:{total_dict['loss']:.5f},acc:{total_dict['acc']:.5f},f1_score:{total_dict['f1_score']:.5f},auc:{total_dict['auc']:.5f}"
        )
        print(
            f"[Val S0] acc:{S0_dict['acc']:.5f},f1_score:{S0_dict['f1_score']:.5f},auc:{S0_dict['auc']:.5f}"
        )
        print(
            f"[Val S1] acc:{S1_dict['acc']:.5f},f1_score:{S1_dict['f1_score']:.5f},auc:{S1_dict['auc']:.5f}"
        )
        print(
            f"[Val S2] acc:{S2_dict['acc']:.5f},f1_score:{S2_dict['f1_score']:.5f},auc:{S2_dict['auc']:.5f}"
        )
        total_timer += timer.elapsed
        result_dict["val_timer"].append(timer.elapsed)

        for prefix, inner_dict in res_dict.items():
            for suffix, val in inner_dict.items():
                key = f"{prefix}_{suffix}"
                if key not in result_dict:
                    result_dict[key] = []
                result_dict[key].append(val)

        metric_value = None
        if metric == "acc":
            metric_value = (S0_dict["acc"] + S2_dict["acc"]) / 2
        elif metric == "auc":
            metric_value = (S0_dict["auc"] + S2_dict["auc"]) / 2
        else:
            metric_value = (S0_dict["f1_score"] + S2_dict["f1_score"]) / 2

        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(metric_value)
        else:
            scheduler.step()
        is_improved = early_stop(metric_value)

        if not is_improved:
            print(
                f"[Early Stop] [Epoch:{current_epoch}/{epochs}] trigger counter: {early_stop.counter}/{early_stop.patience}"
            )
        else:
            torch.save(
                {
                    "epoch": current_epoch,
                    "metric": metric,
                    "metric_average": metric_average,
                    "val_result": res_dict,
                    "encoder": encoder.state_dict(),
                    "classifier": classifier.state_dict(),
                },
                best_save_path,
            )
            print(
                f"[Save Model] [Epoch:{current_epoch}/{epochs}] save best Model successfully"
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

        torch.save(checkpoint, current_save_path)
        print(
            f"[Save Checkpoint] [Epoch:{current_epoch}/{epochs}] save current checkpoint successfully"
        )

        print(f"[Epoch {current_epoch}/{epochs}] total elapsed:{total_timer}")

        pd.DataFrame(result_dict).to_csv(result_dict_path, index=False)

        if early_stop.early_stop:
            print(f"[Early Stop] [Epoch:{current_epoch}/{epochs}] trigger early stop")
            break


def test(config):

    experiment_name = config["name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, _, _, _, test_set, test_itc = load_dataset(**config["dataset"])

    pin_memory = True if torch.cuda.is_available() else False

    loader_config = config["data_loader"]

    test_set_loader = DataLoader(
        test_set,
        collate_fn=test_set.drug_collate_fn,
        pin_memory=pin_memory,
        **loader_config["test_set"],
    )

    test_itc_loader = DataLoader(
        test_itc,
        collate_fn=test_itc.itc_collate_fn,
        pin_memory=pin_memory,
        **loader_config["test_itc"],
    )

    encoder = get_encoder(config["encoder"]).to(device)
    classifier = get_classifier(config["classifier"]).to(device)
    criterion = get_criterion(config["criterion"])

    base_dir = os.path.join(config["save_dir"], experiment_name)
    best_save_path = os.path.join(base_dir, config["best_save_name"])
    test_dict_path = os.path.join(base_dir, config["test_dict_name"])
    record = {"loss": [], "acc": [], "f1_score": [], "auc": []}
    if not os.path.exists(best_save_path):
        print(f"The best model of current experiment:{experiment_name} don't exist")
        return
    print("loading best models")
    best_model = torch.load(best_save_path, weights_only=False)
    encoder.load_state_dict(best_model["encoder"])
    classifier.load_state_dict(best_model["classifier"])
    record["loss"].append(best_model["loss"])
    record["acc"].append(best_model["acc"])
    record["f1_score"].append(best_model["f1_score"])
    record["auc"].append(best_model["auc"])

    print(
        f"[Test Config] Device:{device} Metric:{best_model['metric']} Metric Average:{best_model['metric_average']}"
    )

    print("[Test Start]")

    encoder.eval()
    classifier.eval()

    val_loss = 0.0

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        all_drugs = torch.cat([encoder(drugs.to(device)) for drugs in test_set_loader])

        for d1, d2, labels in test_itc_loader:
            d1, d2, labels = d1.to(device), d2.to(device), labels.to(device)
            logits = classifier(all_drugs[d1], all_drugs[d2])
            loss = criterion(logits, labels)

            preds = torch.argmax(logits, dim=-1)
            prob = torch.softmax(logits, dim=-1)

            val_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(prob.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)
    avg_val_loss = val_loss / len(test_itc_loader)
    val_acc = accuracy_score(all_labels, all_preds)
    val_f1 = f1_score(
        all_labels, all_preds, average=best_model["metric_average"], zero_division=0
    )
    val_auc = roc_auc_score(
        all_labels, all_probs, multi_class="ovr", average=best_model["metric_average"]
    )

    record["loss"].append(avg_val_loss)
    record["acc"].append(val_acc)
    record["f1_score"].append(val_f1)
    record["auc"].append(val_auc)
    print(
        f"[Best] loss:{record['loss'][0]:.5},acc:{record['acc'][0]:.5},f1:{record['f1_score'][0]:.5},auc:{record['auc'][0]:.5}"
    )
    print(
        f"[Test] loss:{record['loss'][1]:.5},acc:{record['acc'][1]:.5},f1:{record['f1_score'][1]:.5},auc:{record['auc'][1]:.5}"
    )
    pd.DataFrame(record).to_csv(test_dict_path, index=False)


def get_configs(config_path):
    configs = None
    if not os.path.exists(config_path):
        print(f"config_path:{config_path} don't exist")
        return None
    if os.path.isfile(config_path):
        configs = [config_path]
    else:
        configs = [
            os.path.join(config_path, config) for config in os.listdir(config_path)
        ]
    return configs


def run_train(config_path):
    configs = get_configs(config_path)
    if configs is None:
        return
    for config in configs:
        with open(config, "r") as f:
            config = json.load(f)
            experiment_name = config["name"]
            print(f"Start Train:{experiment_name}")
            train(config=config)
            print(f"Finish Train:{experiment_name}")


def run_test(config_path):
    configs = get_configs(config_path)
    if configs is None:
        return
    for config in configs:
        with open(config, "r") as f:
            config = json.load(f)
            experiment_name = config["name"]
            print(f"Start Test:{experiment_name}")
            test(config=config)
            print(f"Finish Test:{experiment_name}")


def run(config_path):
    configs = get_configs(config_path)
    if configs is None:
        return
    for config in configs:
        run_train(config)
        run_test(config)
