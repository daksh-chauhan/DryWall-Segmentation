import os
import cv2
import numpy as np
import random

dir = r"datasets\merged_clipseg"
images_dir = os.path.join(dir, "images")
masks_dir = os.path.join(dir, "masks")

images_files = os.listdir(images_dir)
masks_files = os.listdir(masks_dir)

# Shuffle for randomness
combined = list(zip(images_files, masks_files))
random.shuffle(combined)

batch_size = 4
img_display_size = (256, 256)  # resize each image for display
alpha = 0.5  # transparency for overlay

def colorize_mask(mask, color=(255, 0, 0)):
    """Convert grayscale mask to colored mask"""
    colored = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colored[mask > 0] = color
    return colored

def make_horizontal_grid(batch):
    top_row = []
    bottom_row = []
    for img_file, mask_file in batch:
        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)

        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, img_display_size)

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, img_display_size)

        mask_colored = colorize_mask(mask, color=(255, 0, 0))  # red overlay

        # Top row: Photo above mask
        top_combined = np.vstack((img, mask_colored))
        top_row.append(top_combined)

        # Bottom row: Overlay mask with some transparency
        overlay = cv2.addWeighted(img, 1-alpha, mask_colored, alpha, 0)
        bottom_row.append(overlay)

    top_grid = np.hstack(top_row)
    bottom_grid = np.hstack(bottom_row)
    final_grid = np.vstack((top_grid, bottom_grid))
    return final_grid

# Loop through batches
for i in range(0, len(combined), batch_size):
    batch = combined[i:i+batch_size]
    grid = make_horizontal_grid(batch)
    cv2.imshow("Sanity Check - SPACE for next, ESC to exit", cv2.resize(cv2.cvtColor(grid, cv2.COLOR_RGB2BGR),(1024,512)))
    
    key = cv2.waitKey(0)
    if key == 27:  # ESC to exit
        break

cv2.destroyAllWindows()
