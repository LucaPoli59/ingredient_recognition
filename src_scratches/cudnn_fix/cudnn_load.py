import os
# os.environ["LD_LIBRARY_PATH"] = (
#     "/root/miniconda3/envs/wsl_image_pytorch/lib/python3.10/site-packages/nvidia/cudnn/lib:"
#     + os.environ.get("LD_LIBRARY_PATH", "")
# )
print(os.environ["LD_LIBRARY_PATH"])
os.environ["CUDNN_LOGINFO_DBG"] = "1"
os.environ["CUDNN_LOGDEST_DBG"] = "/tmp/cudnn_log.txt"

import torch
print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())

# Check CUDA context initializes at all
torch.cuda.init()
print("CUDA init OK")
torch.cuda.synchronize()
print("CUDA synchronize OK")

t = torch.zeros(1).cuda()
print("Tensor to CUDA OK:", t)

t2 = torch.nn.Conv2d(3, 64, 3, padding=1).cuda()
t_in = torch.randn(1, 3, 32, 32).cuda()
with torch.no_grad():
    out = t2(t_in)
print("Conv2d OK:", out.shape)