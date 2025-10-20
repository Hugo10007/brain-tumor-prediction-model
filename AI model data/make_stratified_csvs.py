# make_stratified_csvs.py
'''
treats every image independently and ensures each class appears in train/val with the same proportions.
'''
import os, glob, pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

ROOT = rf"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\datasets\brain_tumor_dataset\Training"
OUT_TRAIN = os.path.join(ROOT, "labels_train.csv")
OUT_VAL   = os.path.join(ROOT, "labels_val.csv")

classes = ["glioma","meningioma","pituitary","notumor"]
rows = []
for label, cname in enumerate(classes):
    class_dir = os.path.join(ROOT, cname)
    files = glob.glob(os.path.join(class_dir, "*.png")) + glob.glob(os.path.join(class_dir, "*.jpg"))
    files.sort()
    for f in files:
        # patient_id is set to filename prefix (not used for stratified split)
        pid = os.path.splitext(os.path.basename(f))[0]
        rows.append([f, label, pid])

df = pd.DataFrame(rows, columns=["filepath","label","patient_id"])
print("Total images:", len(df))
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
train_idx, val_idx = next(sss.split(df, df["label"]))
train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)

train_df.to_csv(OUT_TRAIN, index=False)
val_df.to_csv(OUT_VAL, index=False)
print("Wrote:", OUT_TRAIN, "(", len(train_df), "rows )")
print("Wrote:", OUT_VAL, "(", len(val_df), "rows )")
print("Val class counts:\n", val_df['label'].value_counts())
