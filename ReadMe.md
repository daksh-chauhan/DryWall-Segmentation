# Dry Wall Segmentation – CLIPSeg Approach

## Overview
This project uses **CLIPSeg**, a text-guided segmentation model, to detect **cracks** and **joins** in drywall surfaces.  
The goal was to evaluate prompt-based segmentation on small, pre-augmented datasets.

---

## Dataset
- Two classes: **Crack** and **Join**  
- Each split (`train`, `val`, `test`) contains:
  - `images/` – input images  
  - `masks/` – binary ground-truth masks  
  - `predictions/` – model outputs  
- Join dataset contains **bounding box annotations** instead of full segmentation masks.

---

## Approach
- Model: **CLIPSeg (CLIPSegForImageSegmentation)**  
- Text prompts used: `"segment crack"` and `"segment taping area"`  
- Inference: sigmoid activation applied on logits to generate binary masks.

---

## Loss Function
- **Binary Cross-Entropy (BCE) Loss**

---

## Metrics
- **IoU (Intersection over Union)**  
- **Dice Score**  
- **Pixel Accuracy**

---

## Outputs
- `split_metrics.xlsx` – per-image metrics for `train`, `val`, `test`  
- `metrics_summary.xlsx` – includes `classwise_metrics` and `overall_metrics`  
- Histograms for IoU, Dice, and Accuracy across each split.

---

## YOLO Plan
A separate YOLO-based segmentation approach was also tested using two specialized models (one for cracks and one for joins).  
The plan was to integrate **text embeddings** and use **cosine similarity** to automatically choose the appropriate model during inference.
