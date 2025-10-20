# evaluate_best.py
import os, numpy as np, pandas as pd
from PIL import Image
import torch
import timm
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

# EDIT if needed
CSV_PATH = r"C:\Users\admin\Desktop\AI COURSEWORK ViT MODEL\Project directory\data\labels.csv"
MODEL_PATH = r"C:\Users\admin\labels.csv\..\best_vit.pth"  # replace with actual best_vit.pth path if different
MODEL_PATH = r"best_vit.pth"  # likely in current working dir
MODEL_NAME = "vit_base_patch16_224"
IMG_SIZE = 224
NUM_CLASSES = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# transforms (must match training)
trans = Compose([Resize(IMG_SIZE, IMG_SIZE), Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)), ToTensorV2()])

# load data
df = pd.read_csv(CSV_PATH)
print("Total rows:", len(df))
print(df['label'].value_counts())

# load model
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, in_chans=3)
ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(ckpt['model_state'])
model.to(DEVICE)
model.eval()

# predict
probs = []
targets = []
filepaths = []
with torch.no_grad():
    for idx, row in df.iterrows():
        img = np.array(Image.open(row.filepath).convert("RGB"))
        a = trans(image=img)
        x = a['image'].unsqueeze(0).to(DEVICE)
        out = model(x)
        p = torch.softmax(out, dim=1).cpu().numpy()[0]
        probs.append(p)
        targets.append(int(row.label))
        filepaths.append(row.filepath)

probs = np.vstack(probs)
targets = np.array(targets)

# slice-level metrics
per_class_auc = []
for c in range(NUM_CLASSES):
    try:
        per_class_auc.append(roc_auc_score((targets==c).astype(int), probs[:,c]))
    except Exception as e:
        per_class_auc.append(float("nan"))
print("Slice-level per-class AUC:", per_class_auc)
print("Slice-level macro AUC:", np.nanmean(per_class_auc))

# confusion & classification report (use argmax)
preds = probs.argmax(axis=1)
print("Confusion matrix (slices):")
print(confusion_matrix(targets, preds))
print("Classification report (slices):")
print(classification_report(targets, preds, digits=4))

# patient-level aggregation (mean probs)
df_probs = df.copy()
df_probs[['p0','p1','p2']] = probs
patient_preds = df_probs.groupby('patient_id')[['p0','p1','p2']].mean()
patient_targets = df_probs.groupby('patient_id')['label'].first().astype(int)

patient_preds_arr = patient_preds.values
patient_targets_arr = patient_targets.values

# patient-level AUC
p_per_class_auc = []
for c in range(NUM_CLASSES):
    try:
        p_per_class_auc.append(roc_auc_score((patient_targets_arr==c).astype(int), patient_preds_arr[:,c]))
    except:
        p_per_class_auc.append(float("nan"))
print("Patient-level per-class AUC:", p_per_class_auc)
print("Patient-level macro AUC:", np.nanmean(p_per_class_auc))

p_preds = patient_preds_arr.argmax(axis=1)
print("Confusion matrix (patients):")
print(confusion_matrix(patient_targets_arr, p_preds))
print("Classification report (patients):")
print(classification_report(patient_targets_arr, p_preds, digits=4))

# Save outputs
df_probs.to_csv("slice_level_probs.csv", index=False)
patient_preds['true_label'] = patient_targets
patient_preds.to_csv("patient_level_probs.csv")
print("Saved slice_level_probs.csv and patient_level_probs.csv")
