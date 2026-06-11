#!/usr/bin/env bash
# Run screening + full normal-group pipeline
PROJ=/home/jason/projects/swingcue-postest
VENV=$PROJ/.venv
CUDA_LIBS="$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VENV/lib/python3.12/site-packages/nvidia/cusparse/lib:$VENV/lib/python3.12/site-packages/nvidia/curand/lib:$VENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
PY="$VENV/bin/python3.12"
cd $PROJ

echo "=== STEP 1: Screening ==="
$PY screen_normal_group.py

echo ""
echo "=== STEP 2: Pipeline for DTL-ready videos ==="
$PY ingest_normal_dtl.py
