#!/usr/bin/env python3
"""probe sam2 availability"""
import sys
try:
    from sam2.build_sam import build_sam2
    print("build_sam2 OK")
except Exception as e:
    print("build_sam2 FAIL:", e)
try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    print("SAM2ImagePredictor OK")
except Exception as e:
    print("SAM2ImagePredictor FAIL:", e)

import os, sam2
p = os.path.dirname(sam2.__file__)
print("sam2 dir:", p)
for x in sorted(os.listdir(p)):
    print(" ", x)
