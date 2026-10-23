"""Generate SAM2 proposals with TF32 disabled for matrix products and convolutions."""
import sys
import torch
from ssrb.cli import main

if __name__ == '__main__':
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    main(['infer-sam2', *sys.argv[1:]])
