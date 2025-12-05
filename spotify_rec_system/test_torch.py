import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
print("Torch imported successfully")
print(f"CUDA available: {torch.cuda.is_available()}")
x = torch.rand(5, 3)
print(x)
