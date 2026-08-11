import os
import json
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc, precision_recall_curve, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle
from torch.optim.lr_scheduler import ReduceLROnPlateau

from model.mobilevit_v2 import create_model, freeze_backbone, unfreeze_finetune_layers, FINETUNE_STAGE

# ----------------------------- REPRODUCIBILITY -----------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ----------------------------- ARGPARSE -----------------------------
parser = argparse.ArgumentParser(description="Train and fine-tune MobileViTv2 for migratory bird classification")
parser.add_argument("--epochs", type=int, required=True, help="Number of epochs for head-only training")
parser.add_argument("--kfolds", type=int, required=True, help="Number of folds for cross-validation")
parser.add_argument("--fine_tune_epochs", type=int, default=10, help="Number of fine-tuning epochs")
args = parser.parse_args()

EPOCHS = args.epochs
K = args.kfolds
FINE_TUNE_EPOCHS = args.fine_tune_epochs

# ----------------------------- FIXED SETTINGS -----------------------------
BATCH_SIZE = 32
HEAD_LR = 1e-4
FINE_TUNE_LR = 2e-5
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4

BASE_DIR = "./"
EXP_NAME = "mobilevitv2_2e-5"
LOG_DIR = os.path.join(BASE_DIR, "logs", EXP_NAME)
WEIGHT_DIR = os.path.join(BASE_DIR, "weights", EXP_NAME)
PLOT_DIR = os.path.join(BASE_DIR, "plots", EXP_NAME)
DATA_DIR = "./data"

os.makedirs(os.path.join(LOG_DIR, "head"), exist_ok=True)
os.makedirs(os.path.join(LOG_DIR, "finetune"), exist_ok=True)
os.makedirs(os.path.join(WEIGHT_DIR, "head"), exist_ok=True)
os.makedirs(os.path.join(WEIGHT_DIR, "finetune"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "head"), exist_ok=True)
os.makedirs(os.path.join(PLOT_DIR, "finetune"), exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = True if torch.cuda.is_available() else False

# ----------------------------- TRANSFORMS -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(256, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.3, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

# ----------------------------- SAFE DATASET -----------------------------
class SafeImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)
        self.skipped_images = 0

    def __getitem__(self, index):
        while True:
            try:
                return super().__getitem__(index)
            except (OSError, ValueError):
                print(f"Skipping corrupted image: {self.samples[index][0]}")
                self.skipped_images += 1
                index = (index + 1) % len(self.samples)

# ----------------------------- PLOTTING -----------------------------
def plot_history(history, fold, classes, phase):
    fold_plot_dir = os.path.join(PLOT_DIR, phase, f"fold{fold}")
    os.makedirs(fold_plot_dir, exist_ok=True)
    sns.set_style("white")
    sns.set_context("paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white'
    })

    pairs = [("train_loss", "val_loss", "Loss"), ("train_accuracy", "val_accuracy", "Accuracy")]
    for m1, m2, label in pairs:
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(history[m1], label=f"Training {label}", color='blue', linewidth=2)
        plt.plot(history[m2], label=f"Validation {label}", color='orange', linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(label, fontsize=12)
        plt.title(f"Fold {fold} {label} ({phase.capitalize()})", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(False)
        plt.savefig(os.path.join(fold_plot_dir, f"{label.lower()}_{phase}_fold{fold}.png"), bbox_inches='tight')
        plt.close()

    for m in ["precision", "recall", "f1"]:
        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(history[m], label=m.capitalize(), color='green', linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(m.capitalize(), fontsize=12)
        plt.title(f"Fold {fold} {m.capitalize()} ({phase.capitalize()})", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(False)
        plt.savefig(os.path.join(fold_plot_dir, f"{m}_{phase}_fold{fold}.png"), bbox_inches='tight')
        plt.close()

    cm = history["confusion_matrix"]
    plt.figure(figsize=(8, 6), dpi=300)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True, linewidths=0)
    plt.title(f"Fold {fold} Confusion Matrix ({phase.capitalize()})", fontsize=14)
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(rotation=0, fontsize=10)
    plt.savefig(os.path.join(fold_plot_dir, f"confusion_matrix_{phase}_fold{fold}.png"), bbox_inches='tight')
    plt.close()

    fpr, tpr, roc_auc = history["roc"]
    plt.figure(figsize=(8, 6), dpi=300)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(len(classes)), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2, label=f'Class {classes[i]} (AUC={roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title(f"Fold {fold} ROC Curves ({phase.capitalize()})", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(False)
    plt.savefig(os.path.join(fold_plot_dir, f"roc_curve_{phase}_fold{fold}.png"), bbox_inches='tight')
    plt.close()

    precision, recall, pr_auc = history["pr"]
    plt.figure(figsize=(8, 6), dpi=300)
    for i, color in zip(range(len(classes)), colors):
        plt.plot(recall[i], precision[i], color=color, lw=2, label=f'Class {classes[i]} (AUC={pr_auc[i]:.2f})')
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title(f"Fold {fold} PR Curves ({phase.capitalize()})", fontsize=14)
    plt.legend(loc="lower left", fontsize=10)
    plt.grid(False)
    plt.savefig(os.path.join(fold_plot_dir, f"pr_curve_{phase}_fold{fold}.png"), bbox_inches='tight')
    plt.close()

def _pad_and_stack(list_of_lists):
    max_len = max(len(l) for l in list_of_lists)
    padded = []
    for l in list_of_lists:
        if len(l) < max_len:
            if len(l) == 0:
                padded.append([0.0]*max_len)
            else:
                padded.append(list(l) + [l[-1]] * (max_len - len(l)))
        else:
            padded.append(list(l))
    return np.vstack(padded)

def plot_overall_metrics(fold_metrics_head, fold_metrics_finetune, classes):
    n_classes = len(classes)
    plot_dir = PLOT_DIR
    os.makedirs(plot_dir, exist_ok=True)
    sns.set_style("white")
    sns.set_context("paper")
    plt.rcParams.update({
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'axes.facecolor': 'white',
        'figure.facecolor': 'white'
    })

    for phase, fold_metrics in [("head", fold_metrics_head), ("finetune", fold_metrics_finetune)]:
        overall_cm = sum(hist["confusion_matrix"] for hist in fold_metrics)
        plt.figure(figsize=(8, 6), dpi=300)
        sns.heatmap(overall_cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes, cbar=True, linewidths=0)
        plt.title(f"Overall Confusion Matrix ({phase.capitalize()})", fontsize=14)
        plt.xlabel("Predicted Label", fontsize=12)
        plt.ylabel("True Label", fontsize=12)
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(rotation=0, fontsize=10)
        plt.savefig(os.path.join(plot_dir, f"confusion_matrix_{phase}_overall.png"), bbox_inches='tight')
        plt.close()

        base_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros((n_classes, 100))
        mean_roc_auc = []
        for i in range(n_classes):
            tpr_list = []
            for hist in fold_metrics:
                fpr, tpr, roc_auc = hist["roc"]
                try:
                    interp_tpr = np.interp(base_fpr, fpr[i], tpr[i])
                except Exception:
                    interp_tpr = np.zeros_like(base_fpr)
                tpr_list.append(interp_tpr)
            mean_tpr[i] = np.mean(tpr_list, axis=0)
            mean_roc_auc.append(auc(base_fpr, mean_tpr[i]))

        plt.figure(figsize=(8, 6), dpi=300)
        colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
        for i, color in zip(range(n_classes), colors):
            plt.plot(base_fpr, mean_tpr[i], color=color, lw=2, label=f'Class {classes[i]} (AUC={mean_roc_auc[i]:.2f})')
        plt.plot([0, 1], [0, 1], 'k--', lw=2)
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title(f"Mean ROC Curves ({phase.capitalize()})", fontsize=14)
        plt.legend(loc="lower right", fontsize=10)
        plt.grid(False)
        plt.savefig(os.path.join(plot_dir, f"roc_curve_{phase}_overall.png"), bbox_inches='tight')
        plt.close()

        base_recall = np.linspace(0, 1, 100)
        mean_precision = np.zeros((n_classes, 100))
        mean_pr_auc = []
        for i in range(n_classes):
            precision_list = []
            for hist in fold_metrics:
                precision, recall, pr_auc = hist["pr"]
                try:
                    interp_precision = np.interp(base_recall, recall[i][::-1], precision[i][::-1])
                except Exception:
                    interp_precision = np.zeros_like(base_recall)
                precision_list.append(interp_precision)
            mean_precision[i] = np.mean(precision_list, axis=0)
            mean_pr_auc.append(auc(base_recall, mean_precision[i]))

        plt.figure(figsize=(8, 6), dpi=300)
        for i, color in zip(range(n_classes), colors):
            plt.plot(base_recall, mean_precision[i], color=color, lw=2, label=f'Class {classes[i]} (AUC={mean_pr_auc[i]:.2f})')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f"Mean PR Curves ({phase.capitalize()})", fontsize=14)
        plt.legend(loc="lower left", fontsize=10)
        plt.grid(False)
        plt.savefig(os.path.join(plot_dir, f"pr_curve_{phase}_overall.png"), bbox_inches='tight')
        plt.close()

    for metric in ["train_loss", "val_loss", "train_accuracy", "val_accuracy"]:
        metric_lists_head = [hist.get(metric, []) for hist in fold_metrics_head]
        metric_lists_finetune = [hist.get(metric, []) for hist in fold_metrics_finetune]
        metric_arr_head = _pad_and_stack(metric_lists_head)
        metric_arr_finetune = _pad_and_stack(metric_lists_finetune)
        mean_metric_head = np.mean(metric_arr_head, axis=0)
        mean_metric_finetune = np.mean(metric_arr_finetune, axis=0)
        epochs_head = np.arange(1, mean_metric_head.shape[0] + 1)
        epochs_finetune = np.arange(1, mean_metric_finetune.shape[0] + 1)

        plt.figure(figsize=(8, 6), dpi=300)
        plt.plot(epochs_head, mean_metric_head, label=f"Head {metric.replace('_', ' ').title()}", color='blue', linewidth=2)
        plt.plot(epochs_finetune, mean_metric_finetune, label=f"Finetune {metric.replace('_', ' ').title()}", color='orange', linewidth=2)
        plt.xlabel("Epoch", fontsize=12)
        plt.ylabel(metric.replace('_', ' ').title(), fontsize=12)
        plt.title(f"Mean {metric.replace('_', ' ').title()} Across Folds (Head vs Finetune)", fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(False)
        plt.savefig(os.path.join(plot_dir, f"{metric}_head_vs_finetune_overall.png"), bbox_inches='tight')
        plt.close()

# ----------------------------- SAVE COMPLEX METRICS -----------------------------
def save_complex_metrics(history, fold, phase, log_dir, classes):
    np.save(os.path.join(log_dir, phase, f"confusion_matrix_{phase}_fold{fold}.npy"), history["confusion_matrix"])
    fpr, tpr, roc_auc = history["roc"]
    precision, recall, pr_auc = history["pr"]
    per_class_metrics = history["per_class_metrics"]
    for i in range(len(classes)):
        np.save(os.path.join(log_dir, phase, f"roc_fpr_{phase}_fold{fold}_class{i}.npy"), fpr[i])
        np.save(os.path.join(log_dir, phase, f"roc_tpr_{phase}_fold{fold}_class{i}.npy"), tpr[i])
        np.save(os.path.join(log_dir, phase, f"roc_auc_{phase}_fold{fold}_class{i}.npy"), np.array(roc_auc[i]))
        np.save(os.path.join(log_dir, phase, f"pr_precision_{phase}_fold{fold}_class{i}.npy"), precision[i])
        np.save(os.path.join(log_dir, phase, f"pr_recall_{phase}_fold{fold}_class{i}.npy"), recall[i])
        np.save(os.path.join(log_dir, phase, f"pr_auc_{phase}_fold{fold}_class{i}.npy"), np.array(pr_auc[i]))
        np.save(os.path.join(log_dir, phase, f"per_class_precision_{phase}_fold{fold}_class{i}.npy"), np.array(per_class_metrics["precision"][i]))
        np.save(os.path.join(log_dir, phase, f"per_class_recall_{phase}_fold{fold}_class{i}.npy"), np.array(per_class_metrics["recall"][i]))
        np.save(os.path.join(log_dir, phase, f"per_class_f1_{phase}_fold{fold}_class{i}.npy"), np.array(per_class_metrics["f1"][i]))

# ----------------------------- MAIN -----------------------------
if __name__ == "__main__":
    set_seed(43)
    dataset = SafeImageFolder(DATA_DIR, transform=train_transform)
    targets = np.array([label for _, label in dataset.samples])
    classes = dataset.classes
    n_classes = len(classes)
    counts = np.bincount(targets, minlength=n_classes).astype(np.float32)
    counts[counts == 0] = 1.0
    class_weights = (targets.shape[0] / (n_classes * counts))
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to(DEVICE)

    dataset_stats = {
        "total_images": len(dataset),
        "num_classes": n_classes,
        "class_counts": {classes[i]: int(counts[i]) for i in range(n_classes)},
        "skipped_images": dataset.skipped_images
    }
    with open(os.path.join(LOG_DIR, "dataset_stats.json"), "w") as f:
        json.dump(dataset_stats, f, indent=4)
    print(f"Dataset Statistics: {dataset_stats}")

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=43)
    fold_metrics_head = []
    fold_metrics_finetune = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(targets)), targets), 1):
        print(f"\nFold {fold}/{K}")
        train_subset = torch.utils.data.Subset(dataset, train_idx)
        val_subset = torch.utils.data.Subset(dataset, val_idx)
        train_subset.dataset.transform = train_transform
        val_subset.dataset.transform = val_transform
        train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                                 num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)
        val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY)

        # Head-only training
        model = create_model(n_classes, DEVICE)
        freeze_backbone(model)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.head.parameters(), lr=HEAD_LR, weight_decay=WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_prec = 0.0
        best_rec = 0.0
        best_f1 = 0.0
        patience = 3
        trigger_times = 0
        best_epoch = 0
        history_head = {
            "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
            "precision": [], "recall": [], "f1": [], "roc": None, "pr": None,
            "confusion_matrix": None, "per_class_metrics": None
        }

        for epoch in range(1, EPOCHS + 1):
            print(f"Head Training Epoch {epoch}/{EPOCHS}")
            model.train()
            running_loss = 0.0
            all_train_preds, all_train_labels = [], []

            for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_train_preds.extend(preds)
                all_train_labels.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            history_head["train_loss"].append(train_loss)
            history_head["train_accuracy"].append(train_acc)

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    preds = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            all_preds = np.vstack(all_preds)
            all_labels = np.concatenate(all_labels)
            pred_classes = np.argmax(all_preds, axis=1)

            acc = accuracy_score(all_labels, pred_classes)
            prec = precision_score(all_labels, pred_classes, average='weighted', zero_division=0)
            rec = recall_score(all_labels, pred_classes, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, pred_classes, average='weighted', zero_division=0)
            per_class_prec = precision_score(all_labels, pred_classes, average=None, zero_division=0)
            per_class_rec = recall_score(all_labels, pred_classes, average=None, zero_division=0)
            per_class_f1 = f1_score(all_labels, pred_classes, average=None, zero_division=0)

            cm = confusion_matrix(all_labels, pred_classes)
            fpr, tpr, roc_auc = {}, {}, {}
            precision, recall, pr_auc = {}, {}, {}
            for i in range(n_classes):
                try:
                    fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_preds[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                except Exception:
                    fpr[i], tpr[i], roc_auc[i] = np.array([0.0]), np.array([0.0]), 0.0
                    print(f"Warning: ROC curve failed for class {classes[i]} in fold {fold} (head)")
                try:
                    precision[i], recall[i], _ = precision_recall_curve(all_labels == i, all_preds[:, i])
                    pr_auc[i] = auc(recall[i], precision[i])
                except Exception:
                    precision[i], recall[i], pr_auc[i] = np.array([0.0]), np.array([0.0]), 0.0
                    print(f"Warning: PR curve failed for class {classes[i]} in fold {fold} (head)")

            history_head["val_loss"].append(val_loss)
            history_head["val_accuracy"].append(acc)
            history_head["precision"].append(prec)
            history_head["recall"].append(rec)
            history_head["f1"].append(f1)
            history_head["confusion_matrix"] = cm
            history_head["roc"] = (fpr, tpr, roc_auc)
            history_head["pr"] = (precision, recall, pr_auc)
            history_head["per_class_metrics"] = {
                "precision": per_class_prec.tolist(),
                "recall": per_class_rec.tolist(),
                "f1": per_class_f1.tolist()
            }

            scheduler.step(acc)

            if acc > best_val_acc:
                best_val_acc = acc
                best_val_loss = val_loss
                best_prec = prec
                best_rec = rec
                best_f1 = f1
                best_epoch = epoch
                trigger_times = 0
                torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "head", f"best_head_model_fold{fold}.pt"))
                print(f"Saved best_head_model_fold{fold}.pt (Val Acc: {best_val_acc:.4f})")
            else:
                trigger_times += 1

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
                break

        history_head["best_val_acc"] = best_val_acc
        history_head["best_prec"] = best_prec
        history_head["best_rec"] = best_rec
        history_head["best_f1"] = best_f1
        with open(os.path.join(LOG_DIR, "head", f"history_head_fold{fold}.json"), "w") as f:
            json.dump({k: v for k, v in history_head.items() if k not in ["roc", "pr", "confusion_matrix", "per_class_metrics"]}, f, indent=4)
        save_complex_metrics(history_head, fold, "head", LOG_DIR, classes)
        plot_history(history_head, fold, classes, "head")
        fold_metrics_head.append(history_head)
        print(f"Saved history, metrics, and plots for head fold {fold}")

        # Fine-tuning
        print(f"\nFine-Tuning Fold {fold}/{K}")
        model.load_state_dict(torch.load(os.path.join(WEIGHT_DIR, "head", f"best_head_model_fold{fold}.pt")))
        unfreeze_finetune_layers(model)
        # NOTE: FINETUNE_STAGE ("layer_5") is MobileViTv2's last backbone stage,
        # the analog of v1's "blocks.4" transformer block.
        transformer_params = [p for n, p in model.named_parameters() if FINETUNE_STAGE in n and p.requires_grad]
        head_params = [p for n, p in model.named_parameters() if "head" in n and p.requires_grad]

        optimizer = torch.optim.AdamW([
            {"params": transformer_params, "lr": FINE_TUNE_LR},
            {"params": head_params, "lr": HEAD_LR},
        ], weight_decay=WEIGHT_DECAY)

        scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2)
        scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

        best_val_acc = 0.0
        best_val_loss = float("inf")
        best_prec = 0.0
        best_rec = 0.0
        best_f1 = 0.0
        trigger_times = 0
        best_epoch = 0
        history_finetune = {
            "train_loss": [], "val_loss": [], "train_accuracy": [], "val_accuracy": [],
            "precision": [], "recall": [], "f1": [], "roc": None, "pr": None,
            "confusion_matrix": None, "per_class_metrics": None
        }

        for epoch in range(1, FINE_TUNE_EPOCHS + 1):
            print(f"Fine-Tuning Epoch {epoch}/{FINE_TUNE_EPOCHS}")
            model.train()
            running_loss = 0.0
            all_train_preds, all_train_labels = [], []

            for imgs, labels in tqdm(train_loader, desc="Training", leave=False):
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                    outputs = model(imgs)
                    loss = criterion(outputs, labels)
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item() * imgs.size(0)
                preds = torch.argmax(outputs, dim=1).cpu().numpy()
                all_train_preds.extend(preds)
                all_train_labels.extend(labels.cpu().numpy())

            train_loss = running_loss / len(train_loader.dataset)
            train_acc = accuracy_score(all_train_labels, all_train_preds)
            history_finetune["train_loss"].append(train_loss)
            history_finetune["train_accuracy"].append(train_acc)

            model.eval()
            val_loss = 0.0
            all_preds, all_labels = [], []

            with torch.no_grad():
                for imgs, labels in tqdm(val_loader, desc="Validating", leave=False):
                    imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=torch.cuda.is_available()):
                        outputs = model(imgs)
                        loss = criterion(outputs, labels)
                    val_loss += loss.item() * imgs.size(0)
                    preds = torch.softmax(outputs, dim=1).cpu().numpy()
                    all_preds.append(preds)
                    all_labels.append(labels.cpu().numpy())

            val_loss /= len(val_loader.dataset)
            all_preds = np.vstack(all_preds)
            all_labels = np.concatenate(all_labels)
            pred_classes = np.argmax(all_preds, axis=1)

            acc = accuracy_score(all_labels, pred_classes)
            prec = precision_score(all_labels, pred_classes, average='weighted', zero_division=0)
            rec = recall_score(all_labels, pred_classes, average='weighted', zero_division=0)
            f1 = f1_score(all_labels, pred_classes, average='weighted', zero_division=0)
            per_class_prec = precision_score(all_labels, pred_classes, average=None, zero_division=0)
            per_class_rec = recall_score(all_labels, pred_classes, average=None, zero_division=0)
            per_class_f1 = f1_score(all_labels, pred_classes, average=None, zero_division=0)

            cm = confusion_matrix(all_labels, pred_classes)
            fpr, tpr, roc_auc = {}, {}, {}
            precision, recall, pr_auc = {}, {}, {}
            for i in range(n_classes):
                try:
                    fpr[i], tpr[i], _ = roc_curve(all_labels == i, all_preds[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                except Exception:
                    fpr[i], tpr[i], roc_auc[i] = np.array([0.0]), np.array([0.0]), 0.0
                    print(f"Warning: ROC curve failed for class {classes[i]} in fold {fold} (finetune)")
                try:
                    precision[i], recall[i], _ = precision_recall_curve(all_labels == i, all_preds[:, i])
                    pr_auc[i] = auc(recall[i], precision[i])
                except Exception:
                    precision[i], recall[i], pr_auc[i] = np.array([0.0]), np.array([0.0]), 0.0
                    print(f"Warning: PR curve failed for class {classes[i]} in fold {fold} (finetune)")

            history_finetune["val_loss"].append(val_loss)
            history_finetune["val_accuracy"].append(acc)
            history_finetune["precision"].append(prec)
            history_finetune["recall"].append(rec)
            history_finetune["f1"].append(f1)
            history_finetune["confusion_matrix"] = cm
            history_finetune["roc"] = (fpr, tpr, roc_auc)
            history_finetune["pr"] = (precision, recall, pr_auc)
            history_finetune["per_class_metrics"] = {
                "precision": per_class_prec.tolist(),
                "recall": per_class_rec.tolist(),
                "f1": per_class_f1.tolist()
            }

            scheduler.step(acc)

            if acc > best_val_acc:
                best_val_acc = acc
                best_val_loss = val_loss
                best_prec = prec
                best_rec = rec
                best_f1 = f1
                best_epoch = epoch
                trigger_times = 0
                torch.save(model.state_dict(), os.path.join(WEIGHT_DIR, "finetune", f"best_finetune_model_fold{fold}.pt"))
                print(f"Saved best_finetune_model_fold{fold}.pt (Val Acc: {best_val_acc:.4f})")
            else:
                trigger_times += 1

            print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                  f"Train Acc: {train_acc:.4f}, Val Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")

            if trigger_times >= patience:
                print(f"Early stopping at epoch {epoch}. Best Val Acc: {best_val_acc:.4f} at epoch {best_epoch}")
                break

        history_finetune["best_val_acc"] = best_val_acc
        history_finetune["best_prec"] = best_prec
        history_finetune["best_rec"] = best_rec
        history_finetune["best_f1"] = best_f1
        with open(os.path.join(LOG_DIR, "finetune", f"history_finetune_fold{fold}.json"), "w") as f:
            json.dump({k: v for k, v in history_finetune.items() if k not in ["roc", "pr", "confusion_matrix", "per_class_metrics"]}, f, indent=4)
        save_complex_metrics(history_finetune, fold, "finetune", LOG_DIR, classes)
        plot_history(history_finetune, fold, classes, "finetune")
        fold_metrics_finetune.append(history_finetune)
        print(f"Saved history, metrics, and plots for finetune fold {fold}")

    # ----------------------------- FINAL SUMMARY & OVERALL PLOTS -----------------------------
    def summarize(metric_name, fold_metrics):
        values = []
        for hist in fold_metrics:
            value = hist.get(f"best_{metric_name}", hist.get(metric_name, [0.0])[-1])
            values.append(value)
        return np.mean(values), np.std(values), values

    mean_acc_head, std_acc_head, acc_head_list = summarize("val_acc", fold_metrics_head)
    mean_prec_head, std_prec_head, prec_head_list = summarize("prec", fold_metrics_head)
    mean_rec_head, std_rec_head, rec_head_list = summarize("rec", fold_metrics_head)
    mean_f1_head, std_f1_head, f1_head_list = summarize("f1", fold_metrics_head)

    mean_acc_finetune, std_acc_finetune, acc_finetune_list = summarize("val_acc", fold_metrics_finetune)
    mean_prec_finetune, std_prec_finetune, prec_finetune_list = summarize("prec", fold_metrics_finetune)
    mean_rec_finetune, std_rec_finetune, rec_finetune_list = summarize("rec", fold_metrics_finetune)
    mean_f1_finetune, std_f1_finetune, f1_finetune_list = summarize("f1", fold_metrics_finetune)

    mean_roc_head = []
    mean_pr_head = []
    mean_roc_finetune = []
    mean_pr_finetune = []
    per_class_metrics_head = {cls: {"precision": [], "recall": [], "f1": []} for cls in classes}
    per_class_metrics_finetune = {cls: {"precision": [], "recall": [], "f1": []} for cls in classes}
    for i, cls in enumerate(classes):
        roc_aucs_head = []
        pr_aucs_head = []
        roc_aucs_finetune = []
        pr_aucs_finetune = []
        for hist in fold_metrics_head:
            try:
                roc_aucs_head.append(hist["roc"][2][i])
                pr_aucs_head.append(hist["pr"][2][i])
                per_class_metrics_head[cls]["precision"].append(hist["per_class_metrics"]["precision"][i])
                per_class_metrics_head[cls]["recall"].append(hist["per_class_metrics"]["recall"][i])
                per_class_metrics_head[cls]["f1"].append(hist["per_class_metrics"]["f1"][i])
            except Exception:
                roc_aucs_head.append(0.0)
                pr_aucs_head.append(0.0)
                per_class_metrics_head[cls]["precision"].append(0.0)
                per_class_metrics_head[cls]["recall"].append(0.0)
                per_class_metrics_head[cls]["f1"].append(0.0)
        for hist in fold_metrics_finetune:
            try:
                roc_aucs_finetune.append(hist["roc"][2][i])
                pr_aucs_finetune.append(hist["pr"][2][i])
                per_class_metrics_finetune[cls]["precision"].append(hist["per_class_metrics"]["precision"][i])
                per_class_metrics_finetune[cls]["recall"].append(hist["per_class_metrics"]["recall"][i])
                per_class_metrics_finetune[cls]["f1"].append(hist["per_class_metrics"]["f1"][i])
            except Exception:
                roc_aucs_finetune.append(0.0)
                pr_aucs_finetune.append(0.0)
                per_class_metrics_finetune[cls]["precision"].append(0.0)
                per_class_metrics_finetune[cls]["recall"].append(0.0)
                per_class_metrics_finetune[cls]["f1"].append(0.0)
        mean_roc_head.append(np.mean(roc_aucs_head))
        mean_pr_head.append(np.mean(pr_aucs_head))
        mean_roc_finetune.append(np.mean(roc_aucs_finetune))
        mean_pr_finetune.append(np.mean(pr_aucs_finetune))

    plot_overall_metrics(fold_metrics_head, fold_metrics_finetune, classes)

    summary_text = "Final Summary:\n\n"
    summary_text += "Per-Fold Validation Metrics (Head-Only):\n"
    summary_text += "\\begin{table}[h]\n\\centering\n\\caption{Per-Fold Validation Metrics (Head-Only)}\n"
    summary_text += "\\begin{tabular}{lcccc}\n\\hline\n"
    summary_text += "Fold & Accuracy & Precision & Recall & F1-Score \\\\ \\hline\n"
    for i in range(K):
        summary_text += f"{i+1} & {acc_head_list[i]*100:.2f}\\% & {prec_head_list[i]*100:.2f}\\% & {rec_head_list[i]*100:.2f}\\% & {f1_head_list[i]*100:.2f}\\% \\\\ \n"
    summary_text += "\\hline\n\\end{tabular}\n\\end{table}\n\n"

    summary_text += "Per-Fold Validation Metrics (Fine-Tuned):\n"
    summary_text += "\\begin{table}[h]\n\\centering\n\\caption{Per-Fold Validation Metrics (Fine-Tuned)}\n"
    summary_text += "\\begin{tabular}{lcccc}\n\\hline\n"
    summary_text += "Fold & Accuracy & Precision & Recall & F1-Score \\\\ \\hline\n"
    for i in range(K):
        summary_text += f"{i+1} & {acc_finetune_list[i]*100:.2f}\\% & {prec_finetune_list[i]*100:.2f}\\% & {rec_finetune_list[i]*100:.2f}\\% & {f1_finetune_list[i]*100:.2f}\\% \\\\ \n"
    summary_text += "\\hline\n\\end{tabular}\n\\end{table}\n\n"

    summary_text += (
        f"Head-Only Validation Metrics across {K} folds:\n"
        f"Accuracy: {mean_acc_head*100:.2f}\\% ± {std_acc_head*100:.2f}\n"
        f"Precision: {mean_prec_head*100:.2f}\\% ± {std_prec_head*100:.2f}\n"
        f"Recall: {mean_rec_head*100:.2f}\\% ± {std_rec_head*100:.2f}\n"
        f"F1-score: {mean_f1_head*100:.2f}\\% ± {std_f1_head*100:.2f}\n"
        f"\nFine-Tuned Validation Metrics across {K} folds:\n"
        f"Accuracy: {mean_acc_finetune*100:.2f}\\% ± {std_acc_finetune*100:.2f}\n"
        f"Precision: {mean_prec_finetune*100:.2f}\\% ± {std_prec_finetune*100:.2f}\n"
        f"Recall: {mean_rec_finetune*100:.2f}\\% ± {std_rec_finetune*100:.2f}\n"
        f"F1-score: {mean_f1_finetune*100:.2f}\\% ± {std_f1_finetune*100:.2f}\n"
        f"\nComparison (Fine-Tuned vs. Head-Only):\n"
        f"Accuracy Improvement: {(mean_acc_finetune - mean_acc_head)*100:.2f}\%\n"
        f"Precision Improvement: {(mean_prec_finetune - mean_prec_head)*100:.2f}\%\n"
        f"Recall Improvement: {(mean_rec_finetune - mean_rec_head)*100:.2f}\%\n"
        f"F1-score Improvement: {(mean_f1_finetune - mean_f1_head)*100:.2f}\%\n"
    )

    summary_text += "\nPer-Class Metrics:\n"
    for i, cls in enumerate(classes):
        mean_prec_head = np.mean(per_class_metrics_head[cls]["precision"])
        mean_rec_head = np.mean(per_class_metrics_head[cls]["recall"])
        mean_f1_head = np.mean(per_class_metrics_head[cls]["f1"])
        mean_prec_finetune = np.mean(per_class_metrics_finetune[cls]["precision"])
        mean_rec_finetune = np.mean(per_class_metrics_finetune[cls]["recall"])
        mean_f1_finetune = np.mean(per_class_metrics_finetune[cls]["f1"])
        summary_text += (
            f"\nClass {cls} (Head-Only):\n"
            f"  Mean ROC AUC: {mean_roc_head[i]*100:.2f}\n"
            f"  Mean PR AUC: {mean_pr_head[i]*100:.2f}\n"
            f"  Mean Precision: {mean_prec_head*100:.2f}\n"
            f"  Mean Recall: {mean_rec_head*100:.2f}\n"
            f"  Mean F1-Score: {mean_f1_head*100:.2f}\n"
            f"Class {cls} (Fine-Tuned):\n"
            f"  Mean ROC AUC: {mean_roc_finetune[i]*100:.2f}\n"
            f"  Mean PR AUC: {mean_pr_finetune[i]*100:.2f}\n"
            f"  Mean Precision: {mean_prec_finetune*100:.2f}\n"
            f"  Mean Recall: {mean_rec_finetune*100:.2f}\n"
            f"  Mean F1-Score: {mean_f1_finetune*100:.2f}\n"
        )

    print("\nFinal Summary:")
    print(summary_text)
    with open(os.path.join(LOG_DIR, "summary.txt"), "w") as f:
        f.write(summary_text)
    print(f"Saved summary.txt in {LOG_DIR}")