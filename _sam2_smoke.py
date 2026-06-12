#!/usr/bin/env python3
"""SAM2 smoke test"""
import torch, numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

cfg  = "configs/sam2.1/sam2.1_hiera_t.yaml"
ckpt = "models/sam2/sam2.1_hiera_tiny.pt"
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

model = build_sam2(cfg, ckpt, device=device)
pred  = SAM2ImagePredictor(model)

img = np.zeros((720, 1280, 3), np.uint8)
pred.set_image(img)
masks, scores, _ = pred.predict(
    point_coords=np.array([[640, 360]]),
    point_labels=np.array([1]),
    multimask_output=False
)
print("mask shape:", masks.shape, "  score:", round(float(scores[0]), 4))
print("SAM2 OK")
