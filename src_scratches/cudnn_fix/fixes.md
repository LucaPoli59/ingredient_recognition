# WSL2 PyTorch/cuDNN Troubleshooting — Fix Summary

## Issue 1: Jupyter Server Failing to Start (PyCharm + WSL2)

**Error:** `Running as root is not recommended. Use --allow-root to bypass.`

**Fix:** Add the following to `~/.jupyter/jupyter_notebook_config.py` (generate it first with `jupyter notebook --generate-config`):

```python
c.ServerApp.allow_root = True
```

---

## Issue 2: Stray Import Shadowing `torch.device`

**Error:** Subtle device placement failures leading to cuDNN errors.

**Fix:** Remove this unused import that was overriding `torch.device`:

```python
# DELETE this line
from torch.xpu import device
```

---

## Issue 3: cuDNN Not Installed in WSL2

**Error:** `CUDNN_STATUS_NOT_INITIALIZED` — `libcudnn.so.9` not found anywhere on the system.

**Fix:** Install cuDNN via pip (it lands under `site-packages/nvidia/cudnn/lib`):

```bash
pip install nvidia-cudnn-cu12
```

---

## Issue 4: cuDNN Installed But Not Found at Runtime

**Error:** `libcudnn.so.9: cannot open shared object file: No such file or directory`

**Cause:** cuDNN 9 was installed under `site-packages/nvidia/cudnn/lib/` but that path was not in `LD_LIBRARY_PATH`.

**Fix:** Add the path via the conda activation script so it applies automatically:

```bash
cat > /root/miniconda3/envs/wsl_image_pytorch/etc/conda/activate.d/cuda_paths.sh << 'EOF'
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.9/lib64:/usr/lib/wsl/lib:$LD_LIBRARY_PATH
EOF
```

Or persist it permanently at the env level:

```bash
conda env config vars set -n wsl_image_pytorch \
  LD_LIBRARY_PATH=/root/miniconda3/envs/wsl_image_pytorch/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.9/lib64:/usr/lib/wsl/lib
```

---

## Issue 5: cuDNN Reporting `compute_capability = 0.0`

**Error:** `cudnnCreate()` failing with `CUDNN_STATUS_NOT_INITIALIZED`, with cuDNN log showing `compute_capability_major: val=0`.

**Cause:** PyTorch `2.8.0+cu129` was too new for the WSL2 GPU driver stack — cuDNN 9.19 couldn't query device architecture through WSL2's `libcuda.so` stub.

**Fix:** Update the Windows NVIDIA GPU drivers to a version that fully supports CUDA 12.9 under WSL2. After the driver update the compute capability was correctly detected and cuDNN initialized successfully.

---

## Issue 6: PyCharm Not Inheriting `LD_LIBRARY_PATH`

**Cause:** PyCharm launches scripts via `conda run` in a fresh shell that doesn't inherit the current terminal's environment.

**Fix (option A):** Use `conda env config vars` — stored in env metadata, always loaded by `conda run`:

```bash
conda env config vars set -n wsl_image_pytorch \
  LD_LIBRARY_PATH=/root/miniconda3/envs/wsl_image_pytorch/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/cuda-12.9/lib64:/usr/lib/wsl/lib

conda deactivate && conda activate wsl_image_pytorch
```

**Fix (option B — fallback):** Set the env var directly in each PyCharm Run Configuration under **Run → Edit Configurations → Environment variables**.
