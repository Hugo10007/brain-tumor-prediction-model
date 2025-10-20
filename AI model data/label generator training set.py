# regen_train_csv.py
import os, glob, pandas as pd

TRAIN_ROOT = rf"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\datasets\brain_tumor_dataset\Training"   # <-- EDIT this to your train root folder
OUT_CSV = os.path.join(TRAIN_ROOT, "labels.csv")

class_names = ["glioma", "meningioma", "pituitary", "notumor"]
class_to_idx = {name:i for i,name in enumerate(class_names)}

rows = []
for cname in class_names:
    class_dir = os.path.join(TRAIN_ROOT, cname)
    if not os.path.isdir(class_dir):
        print("Warning, missing folder:", class_dir)
        continue
    files = glob.glob(os.path.join(class_dir, "*.png")) + glob.glob(os.path.join(class_dir, "*.jpg"))
    files.sort()
    for f in files:
        # patient_id extraction: try to parse 'patient123_img_01.png' or fallback to filename prefix
        name = os.path.basename(f)
        parts = name.split('_')
        if len(parts) >= 2 and parts[0].lower().startswith("patient"):
            patient_id = parts[0]
        elif len(parts) >= 2:
            patient_id = parts[0]
        else:
            patient_id = os.path.splitext(name)[0]
        rows.append([f, class_to_idx[cname], patient_id])

df = pd.DataFrame(rows, columns=["filepath","label","patient_id"])
df.to_csv(OUT_CSV, index=False)
print("Saved train CSV:", OUT_CSV)
print(df['label'].value_counts())
