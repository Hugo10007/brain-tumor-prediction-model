# vit_predict_vscode.py
import os
import sys
import argparse
import numpy as np
from PIL import Image
import torch
import timm
from albumentations import Compose, Resize, Normalize
from albumentations.pytorch import ToTensorV2

import tkinter as tk
from tkinter import filedialog

# --- Edit defaults if needed, I've just set these to temporarily run on my machine & directory ---
DEFAULT_MODEL_PATH = rf"C:\Users\hugop\OneDrive\Desktop\University notes\Year 2\Introduction to Artifical Intelligence\Coursework\AI model data\best_vit.pth"
MODEL_NAME = "vit_base_patch16_224"
IMG_SIZE = 224
NUM_CLASSES = 4
LABEL_NAMES = ["glioma","meningioma","pituitary","notumor"]
# -------------------------------

def build_transforms(img_size=IMG_SIZE):
    return Compose([
        Resize(img_size, img_size),
        Normalize(mean=(0.0,0.0,0.0), std=(1.0,1.0,1.0)),
        ToTensorV2(),
    ])

def load_model(model_path, device):
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES, in_chans=3)
    ckpt = torch.load(model_path, map_location=device)
    # if key 'model_state' exists, use it; otherwise assume full state dict
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state = ckpt['model_state']
    elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    else:
        state = ckpt
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model

def predict_image(model, device, img_path, transforms):
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = Image.open(img_path).convert("RGB")
    arr = np.array(img)
    a = transforms(image=arr)
    x = a['image'].unsqueeze(0).to(device)  # shape (1,3,H,W)
    with torch.no_grad():
        out = model(x)
        probs = torch.softmax(out, dim=1).cpu().numpy()[0]
    return probs

def print_topk(probs, label_names, k=3):
    k = min(k, len(probs))
    order = np.argsort(probs)[::-1][:k]
    for rank, idx in enumerate(order, start=1):
        print(f"{rank}. {label_names[idx]} — prob: {probs[idx]:.4f}")

def choose_file_dialog(title="Select file", filetypes=(("Images","*.png;*.jpg;*.jpeg"),("All files","*.*"))):
    # Use tkinter file dialog if available. On some headless servers tkinter may not be installed.
    try:
        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(title=title, filetypes=(("Image files", "*.png;*.jpg;*.jpeg"), ("All files", "*.*")))
        root.update()
        root.destroy()
        return path
    except Exception:
        return None

def interactive_select_paths(default_model=DEFAULT_MODEL_PATH):
    print("No command-line args provided. Opening file picker(s)...")
    img_path = choose_file_dialog(title="Select MRI image to predict")
    if not img_path:
        img_path = input("Enter image path (or press Enter to cancel): ").strip()
    model_path = None
    # ask if user wants to choose model; default used if they cancel
    use_custom = input(f"Use default model? [{default_model}] (y/n): ").strip().lower()
    if use_custom in ("n", "no"):
        mp = choose_file_dialog(title="Select model checkpoint (best_vit.pth)", filetypes=(("PyTorch","*.pth;*.pt;*.ckpt"),("All files","*.*")))
        if mp:
            model_path = mp
        else:
            model_path = input("Enter model path or press Enter to use default: ").strip() or default_model
    else:
        model_path = default_model
    return img_path, model_path

def main():
    parser = argparse.ArgumentParser(description="Run ViT prediction on one MRI image.")
    parser.add_argument("--image", "-i", help="Path to image to predict (png/jpg). If omitted, a file dialog will open.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL_PATH, help="Path to best_vit.pth checkpoint. If omitted, you can choose interactively.")
    parser.add_argument("--topk", "-k", type=int, default=3, help="Show top-k predictions.")
    args = parser.parse_args()

    # If no CLI image argument provided, open interactive file picker (good for running inside Visual Studio)
    if not args.image:
        img_path, model_path = interactive_select_paths(default_model=args.model)
    else:
        img_path = args.image
        model_path = args.model

    if not img_path:
        print("No image selected. Exiting.")
        return
    if not model_path:
        model_path = DEFAULT_MODEL_PATH

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    print("Loading model from:", model_path)
    try:
        model = load_model(model_path, device)
    except Exception as e:
        print("Error loading model:", e)
        return

    transforms = build_transforms()
    try:
        probs = predict_image(model, device, img_path, transforms)
    except Exception as e:
        print("Error during prediction:", e)
        return

    print("\nImage:", img_path)
    print("\nPredicted probabilities (all classes):")
    for i, name in enumerate(LABEL_NAMES):
        print(f"  {name:12s}: {probs[i]:.4f}")
    print("\nTop predictions:")
    print_topk(probs, LABEL_NAMES, k=args.topk)

    pred_idx = int(np.argmax(probs))
    print(f"\nFinal predicted class: {LABEL_NAMES[pred_idx]} (index {pred_idx})")

if __name__ == "__main__":
    main()
