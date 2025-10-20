# regen_test_csv_from_folders.py
import os, glob, pandas as pd

TEST_ROOT = rf"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\datasets\brain_tumor_dataset\Testing"   # <-- EDIT
OUT_CSV = os.path.join(TEST_ROOT, "labels.csv")  # or test_labels.csv

class_names = ["glioma", "meningioma", "pituitary", "notumor"]
class_to_idx = {name:i for i,name in enumerate(class_names)}

rows=[]
for cname in class_names:
    class_dir = os.path.join(TEST_ROOT, cname)
    if not os.path.isdir(class_dir):
        print("Missing:", class_dir)
        continue
    files = glob.glob(os.path.join(class_dir, "*.png")) + glob.glob(os.path.join(class_dir, "*.jpg"))
    files.sort()
    for f in files:
        patient_id = os.path.splitext(os.path.basename(f))[0]
        rows.append([f, class_to_idx[cname], patient_id])

pd.DataFrame(rows, columns=["filepath","label","patient_id"]).to_csv(OUT_CSV, index=False)
print("Saved test CSV:", OUT_CSV)
