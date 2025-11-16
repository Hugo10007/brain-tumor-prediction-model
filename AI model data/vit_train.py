# vit_train.py
#BY HUGO PIPER

'''

This code is designed to train the Vision transformer on the brain tumour dataset.
It will output a file called best_vit.pth containing the most accurate model data.
best_vit.pth can be tested using vit_predict.py.

'''
import os
import random
import math
from glob import glob
import multiprocessing

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.metrics import roc_auc_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

import pandas as pd
import sys

def check_dataset_distribution(train_df, val_df, test_df):
    """
    Ensures each dataset split has a balanced label distribution
    and all expected classes are present.
    """

    expected_classes = set([0, 1, 2, 3])  # glioma, meningioma, pituitary, no tumor
    splits = {
        "TRAIN": train_df,
        "VAL": val_df,
        "TEST": test_df
    }

    print("\n=== Checking dataset class balance ===")
    all_ok = True

    for name, df in splits.items():
        labels = df['label']
        counts = labels.value_counts().sort_index()

        print(f"\n{name} distribution:")
        print(counts)
        missing = expected_classes - set(counts.index)

        if missing:
            print(f"⚠️  WARNING: Missing classes in {name}: {missing}")
            all_ok = False

        # Detect very small classes (e.g., <10 images)
        small_classes = counts[counts < 10]
        if not small_classes.empty:
            print(f"⚠️  WARNING: {name} has very few samples in classes: {list(small_classes.index)}")
            all_ok = False

    if not all_ok:
        print("\n🚫 ERROR: Dataset splits are unbalanced or missing classes. Please fix your CSVs.")
        print("You can regenerate balanced splits using create_balanced_splits.py.")
        sys.exit(1)
    else:
        print("\n✅ Dataset splits look balanced and ready for training.")


# -------- Config (edit these) --------
# Now expecting explicit train/val/test CSVs (created by your preprocessing scripts)
TRAIN_CSV = r"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\datasets\brain_tumor_dataset\Training\labels_train.csv"
VAL_CSV   = r"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\datasets\brain_tumor_dataset\Training\labels_val.csv"
TEST_CSV  = r"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\datasets\brain_tumor_dataset\Training\labels_test.csv"

train_df = pd.read_csv(TRAIN_CSV)
val_df = pd.read_csv(VAL_CSV)
test_df = pd.read_csv(TEST_CSV)

check_dataset_distribution(train_df, val_df, test_df)


IMG_SIZE = 224
BATCH_SIZE = 16  # adjust to taste / GPU memory
NUM_WORKERS = 4  # 0 is safest on Windows; use >0 for faster IO if stable
NUM_CLASSES = 4  # you have 4 classes
EPOCHS = 6
MODEL_NAME = "vit_base_patch16_224"
LABEL_NAMES = ["glioma","meningioma","pituitary","notumor"]
LR = 5e-5
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE_ES = 4
SEED = 42
N_HEAD_EPOCHS = 2
# ----------------------------------------

# ---------- Model output directory ----------
MODEL_DIR = r"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\AI model data"
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_DIR, "best_vit.pth")
# -------------------------------------------

import pandas as pd
import sys


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class MRIDataset(Dataset):
    def __init__(self, df, transforms=None, in_chans=3):
        self.df = df.reset_index(drop=True)
        self.transforms = transforms
        self.in_chans = in_chans

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = Image.open(row.filepath).convert("RGB")
        img = np.array(img)
        if self.transforms:
            augmented = self.transforms(image=img)
            img = augmented['image']
        label = int(row.label)
        return img, label


def get_transforms(img_size=IMG_SIZE, train=True):
    if train:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.RandomRotate90(),
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.08, rotate_limit=15, p=0.7),
            A.RandomBrightnessContrast(p=0.5),
            A.OneOf([A.GridDistortion(p=0.3), A.ElasticTransform(p=0.3)], p=0.3),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)),
            ToTensorV2(),
        ])


def build_loaders(train_csv=TRAIN_CSV, val_csv=VAL_CSV):
    # read pre-split CSVs (assumes these were produced by your preprocessing)
    train_df = pd.read_csv(train_csv)
    val_df = pd.read_csv(val_csv)
    print("Train CSV rows:", len(train_df))
    print("Val   CSV rows:", len(val_df))
    print("Train head:\n", train_df.head())
    print("Val head:\n", val_df.head())

    train_ds = MRIDataset(train_df, transforms=get_transforms(IMG_SIZE, train=True))
    val_ds = MRIDataset(val_df, transforms=get_transforms(IMG_SIZE, train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE != "cpu"))
    return train_loader, val_loader, train_ds, val_ds, train_df, val_df


def get_model():
    model = timm.create_model(MODEL_NAME, pretrained=True, num_classes=NUM_CLASSES, in_chans=3)
    return model


def get_scheduler(optimizer, train_loader):
    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(0.05 * total_steps)
    from torch.optim.lr_scheduler import LambdaLR

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler):
    model.train()
    running_loss = 0.0
    all_preds, all_targets = [], []
    total_batches = len(train_loader)
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        # use autocast for cuda only; on cpu this is a no-op
        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None and DEVICE.startswith("cuda"))):
            outputs = model(imgs)
            loss = criterion(outputs, labels)
        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if scheduler:
            scheduler.step()

        running_loss += loss.item() * imgs.size(0)
        preds = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        all_preds.append(preds)
        all_targets.append(labels.detach().cpu().numpy())

    epoch_loss = running_loss / (len(train_loader.dataset) if hasattr(train_loader, "dataset") else 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    try:
        aucs = []
        for c in range(NUM_CLASSES):
            aucs.append(roc_auc_score((all_targets == c).astype(int), all_preds[:, c]))
        mean_auc = np.mean(aucs)
    except Exception:
        mean_auc = float("nan")
    return epoch_loss, mean_auc


def validate(model, val_loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_targets = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * imgs.size(0)
            preds = torch.softmax(outputs, dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(labels.cpu().numpy())
    epoch_loss = running_loss / (len(val_loader.dataset) if hasattr(val_loader, "dataset") else 1)
    all_preds = np.concatenate(all_preds, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    aucs = []
    for c in range(NUM_CLASSES):
        try:
            aucs.append(roc_auc_score((all_targets == c).astype(int), all_preds[:, c]))
        except Exception:
            aucs.append(float("nan"))
    mean_auc = np.nanmean(aucs)
    return epoch_loss, mean_auc, all_preds, all_targets


def evaluate_test(model_path=SAVE_PATH, test_csv=TEST_CSV):
    # Optional helper to evaluate the saved best checkpoint on the test CSV
    if not os.path.exists(test_csv):
        print("No test CSV found at", test_csv, "- skipping test evaluation.")
        return
    print("Evaluating saved model on test CSV:", test_csv)
    device = DEVICE
    model = get_model().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get('model_state', ckpt)
    model.load_state_dict(state)
    model.eval()

    import numpy as np
    from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
    from albumentations import Compose, Resize, Normalize
    from albumentations.pytorch import ToTensorV2

    trans = Compose([Resize(IMG_SIZE, IMG_SIZE), Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0)), ToTensorV2()])
    df = pd.read_csv(test_csv)
    probs = []
    preds = []
    targets = []
    with torch.no_grad():
        for _, row in df.iterrows():
            fp = row.filepath
            img = Image.open(fp).convert("RGB")
            a = trans(image=np.array(img))
            x = a['image'].unsqueeze(0).to(device)
            out = model(x)
            p = torch.softmax(out, dim=1).cpu().numpy()[0]
            probs.append(p)
            preds.append(int(p.argmax()))
            targets.append(int(row.label))
    probs = np.vstack(probs)
    targets = np.array(targets)
    preds = np.array(preds)

    per_class_auc = []
    for c in range(NUM_CLASSES):
        try:
            per_class_auc.append(roc_auc_score((targets == c).astype(int), probs[:, c]))
        except Exception:
            per_class_auc.append(float("nan"))
    print("Per-class AUC (test):", per_class_auc)
    print("Macro AUC (test):", np.nanmean(per_class_auc))
    print("Confusion matrix (test):")
    print(confusion_matrix(targets, preds))
    print("Classification report (test):")
    print(classification_report(targets, preds, digits=4))


def run_training():
    seed_everything()
    print("Device:", DEVICE)
    train_loader, val_loader, train_ds, val_ds, train_df, val_df = build_loaders(TRAIN_CSV, VAL_CSV)

    model = get_model().to(DEVICE)
    print("Model loaded. Params:", sum(p.numel() for p in model.parameters()))

    # Compute class weights from the training split and create weighted loss
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.unique(train_df['label'])
    weights = compute_class_weight('balanced', classes=classes, y=train_df['label'].values)
    # Map weights into a vector of length NUM_CLASSES (ensures correct ordering)
    weight_vec = np.ones(NUM_CLASSES, dtype=np.float32)
    for i, c in enumerate(classes):
        weight_vec[int(c)] = weights[i]
    class_weights = torch.tensor(weight_vec, dtype=torch.float).to(DEVICE)
    print("Class weights:", class_weights.cpu().numpy())
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Freeze backbone, train head only for first N_HEAD_EPOCHS
    for name, p in model.named_parameters():
        p.requires_grad = False
    if hasattr(model, 'head'):
        for p in model.head.parameters():
            p.requires_grad = True
    else:
        # fallback: unfreeze last few parameters if head is named differently
        for name, p in list(model.named_parameters())[-10:]:
            p.requires_grad = True

    # create optimizer only for params that require_grad
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = get_scheduler(optimizer, train_loader)
    scaler = torch.amp.GradScaler() if DEVICE.startswith("cuda") else None

    best_auc = 0.0
    no_improve = 0
    history = []
    for epoch in range(1, EPOCHS + 1):
        # If we've finished the head-only warmup epochs, unfreeze everything and re-create optimizer/scheduler
        if epoch == N_HEAD_EPOCHS + 1:
            print("Unfreezing entire model and re-creating optimizer for full fine-tuning.")
            for p in model.parameters():
                p.requires_grad = True
            optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            scheduler = get_scheduler(optimizer, train_loader)

        train_loss, train_auc = train_one_epoch(model, train_loader, optimizer, criterion, scheduler, scaler)
        val_loss, val_auc, val_preds, val_targets = validate(model, val_loader, criterion)
        print(f"Epoch {epoch}/{EPOCHS}  Train loss: {train_loss:.4f} AUC: {train_auc:.4f}  Val loss: {val_loss:.4f} AUC: {val_auc:.4f}")

        history.append({'epoch': epoch, 'train_loss': train_loss, 'train_auc': train_auc, 'val_loss': val_loss, 'val_auc': val_auc})
        pd.DataFrame(history).to_csv(os.path.join(MODEL_DIR, "history.csv"), index=False)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'model_state': model.state_dict(), 'optimizer': optimizer.state_dict()}, SAVE_PATH)
            print(f"Saved best model to: {SAVE_PATH}")
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= PATIENCE_ES:
            print("Early stopping.")
            break
    print("Training finished. Best val AUC:", best_auc)

    # run test evaluation if test csv exists
    if os.path.exists(TEST_CSV):
        evaluate_test(SAVE_PATH, TEST_CSV)


if __name__ == "__main__":
    # Required on Windows when using multiprocessing with DataLoader
    multiprocessing.freeze_support()
    run_training()
