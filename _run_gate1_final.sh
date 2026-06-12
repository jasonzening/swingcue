#!/usr/bin/env bash
# Gate-1 final + pipeline first-loop
PROJ=/home/jason/projects/swingcue-postest
VENV=$PROJ/.venv
CUDA_LIBS="$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VENV/lib/python3.12/site-packages/nvidia/cusparse/lib:$VENV/lib/python3.12/site-packages/nvidia/curand/lib:$VENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
PY="$VENV/bin/python3.12"

cd $PROJ

echo "=== GATE-1 FINAL v3.2 ==="
$PY run_e1.py

echo ""
echo "=== PIPELINE: 201054 DTL ==="
$PY run_pipeline.py \
    input/Videos2026-06-09_201054_561.mp4 \
    --angle down-the-line \
    --gt_impact 150 \
    --out pipeline_output

echo ""
echo "=== PIPELINE: 201058 DTL ==="
$PY run_pipeline.py \
    input/Videos2026-06-09_201058_697.mp4 \
    --angle down-the-line \
    --gt_impact 186 \
    --out pipeline_output

echo ""
echo "ALL DONE"
