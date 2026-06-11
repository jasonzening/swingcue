#!/usr/bin/env bash
PROJ=/home/jason/projects/swingcue-postest
VENV=$PROJ/.venv
CUDA_LIBS="$VENV/lib/python3.12/site-packages/nvidia/cuda_runtime/lib:$VENV/lib/python3.12/site-packages/nvidia/cudnn/lib:$VENV/lib/python3.12/site-packages/nvidia/cublas/lib:$VENV/lib/python3.12/site-packages/nvidia/cufft/lib:$VENV/lib/python3.12/site-packages/nvidia/cusparse/lib:$VENV/lib/python3.12/site-packages/nvidia/curand/lib:$VENV/lib/python3.12/site-packages/nvidia/cusolver/lib:$VENV/lib/python3.12/site-packages/nvidia/nvjitlink/lib:/usr/lib/wsl/lib"
export LD_LIBRARY_PATH="$CUDA_LIBS:${LD_LIBRARY_PATH:-}"
cd $PROJ
"$VENV/bin/python3.12" batch2_pipeline.py
