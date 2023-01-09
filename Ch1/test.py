import os
import torch

print("Pytorch can access the GPU: {0} ".format(True if torch.cuda.is_available() else False))
print(os.system("nvidia-smi"))