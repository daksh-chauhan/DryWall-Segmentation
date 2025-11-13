import os
import time
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

# ==========================
# CONFIG
# ==========================
base_dir = r"F:\Projects\Dry_Wall_Segmentation\datasets\merged_clipseg\splits"
model_path = r"C:\Users\daksh\Downloads\clipseg_best-20251112T123318Z-1-001\clipseg_best"
device = "cuda" if torch.cuda.is_available() else "cpu"

results_dir = os.path.join(base_dir, "results")
os.makedirs(results_dir, exist_ok=True)

processor = CLIPSegProcessor.from_pretrained(model_path)
model = CLIPSegForImageSegmentation.from_pretrained(model_path)
model.to(device).eval()

# ==========================
# METRICS
# ==========================
def iou_score(pred, target):
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    return intersection / (union + 1e-8)

def dice_score(pred, target):
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    intersection = np.logical_and(pred, target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-8)

def pixel_accuracy(pred, target):
    pred = (pred > 0.5).astype(np.uint8)
    target = (target > 0.5).astype(np.uint8)
    correct = (pred == target).sum()
    total = np.prod(pred.shape)
    return correct / total

# ==========================
# LOOP OVER SPLITS
# ==========================
splits = ["train", "val", "test"]
all_results = []
split_dfs = {}

for split in splits:
    split_path = os.path.join(base_dir, split)
    image_dir = os.path.join(split_path, "images")
    mask_dir = os.path.join(split_path, "masks")
    pred_dir = os.path.join(split_path, "predictions")
    os.makedirs(pred_dir, exist_ok=True)

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    split_results = []

    for filename in tqdm(image_files, desc=f"Evaluating {split}"):
        img_path = os.path.join(image_dir, filename)
        mask_path = os.path.join(mask_dir, os.path.splitext(filename)[0] + ".png")
        pred_path = os.path.join(pred_dir, os.path.splitext(filename)[0] + ".png")

        # --- If prediction missing, infer ---
        if not os.path.exists(pred_path):
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                print(f"Skipping unreadable image: {filename}")
                continue

            if 'join' in filename.lower():
                prompt = "segment taping area"
            elif 'crack' in filename.lower():
                prompt = "segment crack"
            else:
                prompt = "segment damaged area"

            inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
                preds = torch.sigmoid(outputs.logits)

            pred_mask_resized = F.interpolate(
                preds.unsqueeze(1) if preds.dim() == 3 else preds,
                size=(image.height, image.width),
                mode="bilinear",
                align_corners=False
            )[0, 0].cpu().numpy()

            Image.fromarray((pred_mask_resized * 255).astype(np.uint8)).save(pred_path)
        else:
            pred_mask_resized = np.array(Image.open(pred_path).convert("L")) / 255.0

        if not os.path.exists(mask_path):
            print(f"No ground truth for {filename}")
            continue

        gt_mask = np.array(Image.open(mask_path).convert("L")) / 255.0

        # --- Metrics ---
        iou = iou_score(pred_mask_resized, gt_mask)
        dice = dice_score(pred_mask_resized, gt_mask)
        acc = pixel_accuracy(pred_mask_resized, gt_mask)

        class_label = "crack" if "crack" in filename.lower() else "join"

        split_results.append({
            "split": split,
            "class": class_label,
            "filename": filename,
            "iou": iou,
            "dice": dice,
            "pixel_acc": acc
        })

    df = pd.DataFrame(split_results)
    split_dfs[split] = df  # store for Excel later

    # --- Aggregate per class ---
    class_groups = df.groupby("class").agg(
        mean_iou=('iou', 'mean'),
        mean_dice=('dice', 'mean'),
        mean_acc=('pixel_acc', 'mean'),
        count=('filename', 'count')
    ).reset_index()

    class_groups["split"] = split
    all_results.append(class_groups)

    # --- Histograms ---
    for metric in ['iou', 'dice', 'pixel_acc']:
        plt.figure(figsize=(6, 4))
        plt.hist(df[metric].dropna(), bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f"{split.upper()} — {metric.upper()} distribution")
        plt.xlabel(metric)
        plt.ylabel("Frequency")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{split}_{metric}_hist.png"))
        plt.close()

# ==========================
# FINAL AGGREGATES + EXCEL EXPORT
# ==========================
final_df = pd.concat(all_results, ignore_index=True)

# Mean across classes → mIoU etc.
overall = final_df.groupby("split").agg(
    mIoU=('mean_iou', 'mean'),
    mDice=('mean_dice', 'mean'),
    mAcc=('mean_acc', 'mean')
).reset_index()

# ---- Excel File A: Summary ----
summary_path = os.path.join(results_dir, "metrics_summary.xlsx")
with pd.ExcelWriter(summary_path, engine='openpyxl') as writer:
    final_df.to_excel(writer, sheet_name='classwise_metrics', index=False)
    overall.to_excel(writer, sheet_name='overall_metrics', index=False)

# ---- Excel File B: Per-Split ----
split_path = os.path.join(results_dir, "split_metrics.xlsx")
with pd.ExcelWriter(split_path, engine='openpyxl') as writer:
    for split, df in split_dfs.items():
        df.to_excel(writer, sheet_name=split, index=False)

print("\n===============================")
print("✅ Evaluation complete! Excel files created.")
print("===============================")
print(final_df)
print("\n--- Overall Mean ---")
print(overall)
print(f"\n📁 All Excel results saved in: {results_dir}")
