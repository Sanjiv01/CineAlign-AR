"""
Environment verification script for CineAlign-AR.
Checks all critical dependencies, GPU access, and model availability.

Usage:
    python test_env.py           # Quick check (GPU + core imports)
    python test_env.py --full    # Full check (all models + inference test)
"""

import sys
import os

PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
WARN = "\033[93m[WARN]\033[0m"
INFO = "\033[94m[INFO]\033[0m"

errors = []
warnings = []


def check(name, fn):
    """Run a check and report pass/fail."""
    try:
        result = fn()
        print(f"  {PASS} {name}: {result}")
        return True
    except Exception as e:
        print(f"  {FAIL} {name}: {e}")
        errors.append(f"{name}: {e}")
        return False


def check_warn(name, fn):
    """Run a check; warn on failure instead of error."""
    try:
        result = fn()
        print(f"  {PASS} {name}: {result}")
        return True
    except Exception as e:
        print(f"  {WARN} {name}: {e}")
        warnings.append(f"{name}: {e}")
        return False


# ============================================================
# 1. CORE PYTHON & GPU
# ============================================================
print(f"\n{'='*60}")
print(f"{INFO} 1. Core Python & GPU")
print(f"{'='*60}")

check("Python version", lambda: sys.version.split()[0])

def check_torch():
    import torch
    ver = torch.__version__
    cuda = torch.version.cuda or "None"
    return f"{ver} (CUDA {cuda})"
check("PyTorch", check_torch)

def check_gpu():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    count = torch.cuda.device_count()
    names = [torch.cuda.get_device_name(i) for i in range(count)]
    return f"{count} GPU(s): {', '.join(names)}"
check("GPU access", check_gpu)

def check_gpu_memory():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU")
    total = torch.cuda.get_device_properties(0).total_mem / (1024**3)
    return f"{total:.1f} GB on GPU 0"
check("GPU memory", check_gpu_memory)

# ============================================================
# 2. ML FRAMEWORKS
# ============================================================
print(f"\n{'='*60}")
print(f"{INFO} 2. ML Frameworks")
print(f"{'='*60}")

check("transformers", lambda: __import__("transformers").__version__)
check("diffusers", lambda: __import__("diffusers").__version__)
check("accelerate", lambda: __import__("accelerate").__version__)
check("peft", lambda: __import__("peft").__version__)
check("safetensors", lambda: __import__("safetensors").__version__)
check_warn("flash_attn", lambda: __import__("flash_attn").__version__)

# ============================================================
# 3. VIDEO PROCESSING
# ============================================================
print(f"\n{'='*60}")
print(f"{INFO} 3. Video Processing")
print(f"{'='*60}")

check("opencv (cv2)", lambda: __import__("cv2").__version__)
check("decord", lambda: __import__("decord").__version__)
check("scenedetect", lambda: __import__("scenedetect").__version__)
check_warn("imageio", lambda: __import__("imageio").__version__)

def check_ffmpeg():
    import subprocess
    result = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
    first_line = result.stdout.split('\n')[0] if result.stdout else "unknown"
    return first_line.strip()
check_warn("ffmpeg (system)", check_ffmpeg)

# ============================================================
# 4. DETECTION & FACE MODELS
# ============================================================
print(f"\n{'='*60}")
print(f"{INFO} 4. Detection & Face Recognition")
print(f"{'='*60}")

check("ultralytics (YOLO)", lambda: __import__("ultralytics").__version__)
check("insightface", lambda: __import__("insightface").__version__)
check("onnxruntime", lambda: __import__("onnxruntime").__version__)
check("sklearn", lambda: __import__("sklearn").__version__)

# ============================================================
# 5. UTILITY PACKAGES
# ============================================================
print(f"\n{'='*60}")
print(f"{INFO} 5. Utility Packages")
print(f"{'='*60}")

check("numpy", lambda: __import__("numpy").__version__)
check("pandas", lambda: __import__("pandas").__version__)
check("PIL/Pillow", lambda: __import__("PIL").__version__)
check("tqdm", lambda: __import__("tqdm").__version__)
check("easydict", lambda: __import__("easydict").__version__)
check("einops", lambda: __import__("einops").__version__)
check_warn("huggingface_hub", lambda: __import__("huggingface_hub").__version__)

# ============================================================
# 6. MODEL WEIGHTS (optional --full check)
# ============================================================
if "--full" in sys.argv:
    print(f"\n{'='*60}")
    print(f"{INFO} 6. Model Weights")
    print(f"{'='*60}")

    project_root = os.path.dirname(os.path.abspath(__file__))

    def check_wan21():
        p = os.path.join(project_root, "models", "Wan2.1-T2V-1.3B")
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Not found: {p}")
        files = os.listdir(p)
        return f"{len(files)} files in {p}"
    check("Wan2.1-T2V-1.3B", check_wan21)

    def check_echoshot():
        p = os.path.join(project_root, "models", "EchoShot")
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Not found: {p}")
        files = os.listdir(p)
        return f"{len(files)} files in {p}"
    check("EchoShot weights", check_echoshot)

    def check_yolo_weights():
        p1 = os.path.join(project_root, "models", "yolov8m.pt")
        p2 = "yolov8m.pt"
        if os.path.isfile(p1):
            size_mb = os.path.getsize(p1) / (1024*1024)
            return f"{size_mb:.1f} MB at {p1}"
        elif os.path.isfile(p2):
            size_mb = os.path.getsize(p2) / (1024*1024)
            return f"{size_mb:.1f} MB at {p2}"
        raise FileNotFoundError("yolov8m.pt not found in models/ or current dir")
    check_warn("YOLOv8m weights", check_yolo_weights)

    def check_insightface_model():
        home = os.path.expanduser("~")
        p = os.path.join(home, ".insightface", "models", "buffalo_l")
        if not os.path.isdir(p):
            raise FileNotFoundError(f"Not found: {p}")
        files = os.listdir(p)
        return f"{len(files)} files in {p}"
    check_warn("InsightFace buffalo_l", check_insightface_model)

    # Quick inference sanity test
    print(f"\n{'='*60}")
    print(f"{INFO} 7. Quick Inference Sanity Test")
    print(f"{'='*60}")

    def check_torch_matmul():
        import torch
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        c = torch.mm(a, b)
        torch.cuda.synchronize()
        return f"256x256 matmul OK, result norm = {c.norm().item():.2f}"
    check("CUDA matmul", check_torch_matmul)

# ============================================================
# SUMMARY
# ============================================================
print(f"\n{'='*60}")
if errors:
    print(f"{FAIL} {len(errors)} error(s), {len(warnings)} warning(s)")
    for e in errors:
        print(f"  ERROR: {e}")
    for w in warnings:
        print(f"  WARN:  {w}")
    sys.exit(1)
elif warnings:
    print(f"{WARN} All critical checks passed, {len(warnings)} warning(s)")
    for w in warnings:
        print(f"  WARN:  {w}")
else:
    print(f"{PASS} All checks passed!")
print(f"{'='*60}\n")
