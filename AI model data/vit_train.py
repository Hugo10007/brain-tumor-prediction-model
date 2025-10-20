# vit_train.py (Windows-safe with main guard)
import os
import random
import math
from glob import glob
from collections import defaultdict
import multiprocessing

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import timm

from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score

import albumentations as A
from albumentations.pytorch import ToTensorV2

# -------- Config (edit these) --------
DATA_CSV = rf"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\datasets\brain_tumor_dataset\Training\labels.csv"
IMG_SIZE = 224
BATCH_SIZE = 16 #My GPU (RTX 3070) runs well using a batch size of 16 but weaker gpus may need a batch size of 8 or 4
NUM_WORKERS = 0  # Set to 0 for quick test without multiprocessing
NUM_CLASSES = min(4, os.cpu_count()-1)
EPOCHS = 6  #CHANGE THIS BACK TO 6, 1 IS FOR TRAINING PURPOSES ONLY
MODEL_NAME = "vit_base_patch16_224"
LABEL_NAMES = ["glioma","meningioma","pituitary","notumor"]
LR = 5e-5
WEIGHT_DECAY = 1e-2
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PATIENCE_ES = 4
SEED = 42
N_HEAD_EPOCHS = 2

# ---------- Model output directory ----------
#This is where best_vit.pth gets saved. best_vit.pth is the model training data so the model doesnt need to retrain after every use.
MODEL_DIR = rf"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\AI model data"
os.makedirs(MODEL_DIR, exist_ok=True)
SAVE_PATH = os.path.join(MODEL_DIR, "best_vit.pth")
# -------------------------------------------


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
            A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
            ToTensorV2(),
        ])

def build_loaders(csv_path=DATA_CSV):
    df = pd.read_csv(csv_path)
    # Basic check
    print("CSV rows:", len(df))
    print(df.head())

    # patient-level split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
    train_idx, val_idx = next(gss.split(df, df['label'], groups=df['patient_id']))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)
    print("Train samples:", len(train_df), "Val samples:", len(val_df))

    train_ds = MRIDataset(train_df, transforms=get_transforms(IMG_SIZE, train=True))
    val_ds = MRIDataset(val_df, transforms=get_transforms(IMG_SIZE, train=False))

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=(DEVICE!="cpu"))
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=(DEVICE!="cpu"))
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
    for imgs, labels in train_loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
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
            aucs.append(roc_auc_score((all_targets==c).astype(int), all_preds[:,c]))
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
            aucs.append(roc_auc_score((all_targets==c).astype(int), all_preds[:,c]))
        except Exception:
            aucs.append(float("nan"))
    mean_auc = np.nanmean(aucs)
    return epoch_loss, mean_auc, all_preds, all_targets

def run_training():
    seed_everything()
    print("Device:", DEVICE)
    # build loaders and get train_df for class weights
    train_loader, val_loader, train_ds, val_ds, train_df, val_df = build_loaders(DATA_CSV)

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
    scaler = torch.cuda.amp.GradScaler() if DEVICE.startswith("cuda") else None

    best_auc = 0.0
    no_improve = 0
    for epoch in range(1, EPOCHS+1):
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

if __name__ == "__main__":
    # Required on Windows when using multiprocessing with DataLoader
    multiprocessing.freeze_support()
    run_training()
