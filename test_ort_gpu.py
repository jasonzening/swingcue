import onnxruntime as ort
import numpy as np
import subprocess
import os

print("ORT version:", ort.__version__)
print("Available providers:", ort.get_available_providers())

# Check CUDA libs in ldconfig
try:
    r = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True)
    cuda_libs = [l for l in r.stdout.split("\n") if any(x in l for x in ["libcuda", "libcudart", "libcudnn"])]
    print("\nCUDA libs in ldconfig:")
    for l in cuda_libs[:10]:
        print(" ", l.strip())
except Exception as e:
    print("ldconfig error:", e)

# Check LD_LIBRARY_PATH
print("\nLD_LIBRARY_PATH:", os.environ.get("LD_LIBRARY_PATH", "(not set)"))
print("PATH snippet:", os.environ.get("PATH", "")[:200])

# Check if libcuda.so.1 is findable
try:
    import ctypes
    libcuda = ctypes.CDLL("libcuda.so.1")
    print("\nlibcuda.so.1: FOUND via ctypes")
except OSError as e:
    print("\nlibcuda.so.1: NOT FOUND -", e)

try:
    libcudart = ctypes.CDLL("libcudart.so.12")
    print("libcudart.so.12: FOUND via ctypes")
except OSError as e:
    print("libcudart.so.12: NOT FOUND -", e)

# Try to create a real ORT session with CUDA
# Use a tiny dummy ONNX model
try:
    import io
    # Create a tiny identity ONNX model
    import onnx
    from onnx import helper, TensorProto
    X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1, 3])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, 3])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [X], [Y])
    model = helper.make_model(graph)
    model_bytes = model.SerializeToString()
    
    sess = ort.InferenceSession(
        model_bytes,
        providers=[("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
    )
    print("\nDummy ORT session created, provider:", sess.get_providers())
    x = np.ones((1, 3), dtype=np.float32)
    out = sess.run(None, {"X": x})
    print("Inference OK, output:", out)
except ImportError:
    print("\nonnx not installed for dummy model test")
except Exception as e:
    print("\nORT CUDA session error:", e)
