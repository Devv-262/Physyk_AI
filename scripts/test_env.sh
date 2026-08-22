#!/bin/bash
set -e
echo "=== Testing Python and Torch in Isaac Sim 6.0.1 ==="
docker run --rm --gpus all -e "ACCEPT_EULA=Y" nvcr.io/nvidia/isaac-sim:6.0.1 ./python.sh -c "import torch; print('PyTorch Version:', torch.__version__, '| CUDA Available:', torch.cuda.is_available(), '| GPU:', torch.cuda.get_device_name(0))"
