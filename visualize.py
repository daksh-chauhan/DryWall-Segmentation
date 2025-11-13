import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# Paths
for split in ['train', 'val', 'test']:
    base = fr"F:\Projects\Dry_Wall_Segmentation\datasets\merged_clipseg\splits\{split}"

    img_dir = os.path.join(base, "images")
    mask_dir = os.path.join(base, "masks")
    pred_dir = os.path.join(base, "predictions")

    # Create output folder for visualizations
    out_dir = os.path.join(base, "visualizations")
    os.makedirs(out_dir, exist_ok=True)

    # Collect image names
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for name in tqdm(img_files):
        basename = os.path.splitext(name)[0]

        # Infer paths
        img_path = os.path.join(img_dir, name)
        mask_path = os.path.join(mask_dir, basename + ".png")
        pred_path = os.path.join(pred_dir, basename + ".png")

        # Skip if missing any file
        if not (os.path.exists(mask_path) and os.path.exists(pred_path)):
            print(f"Skipping {basename}: missing mask or prediction.")
            continue

        # Read images
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

        # Resize all to match
        h, w = image.shape[:2]
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)

        # Overlay predicted mask (colored, translucent)
        colored_pred = cv2.applyColorMap(pred, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image, 0.6, colored_pred, 0.4, 0)

        # Detect crack/join from filename
        label = "crack" if "crack" in basename.lower() else ("join" if "join" in basename.lower() else "unknown")

        # Create figure
        fig, axs = plt.subplots(2, 2, figsize=(8, 8))
        fig.suptitle(label.upper(), color='orange', fontsize=14, fontweight='bold', y=0.98)

        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Image")

        axs[0, 1].imshow(overlay)
        axs[0, 1].set_title("Image + Pred Mask")

        axs[1, 0].imshow(mask, cmap='gray')
        axs[1, 0].set_title("Ground Truth Mask")

        axs[1, 1].imshow(pred, cmap='gray')
        axs[1, 1].set_title("Predicted Mask")

        for ax in axs.flat:
            ax.axis('off')

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.subplots_adjust(wspace=0.05, hspace=0.1)

        # Save result
        out_path = os.path.join(out_dir, basename + "_viz.png")
        plt.savefig(out_path)
        plt.close(fig)

        # print(f"Saved visualization for {basename}")

    print("✅ All visualizations done!")
