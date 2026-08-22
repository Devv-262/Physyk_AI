#!/bin/bash
# `vllm` isn't on PATH by default in a plain shell — it lives in the .venv-nemotron
# virtualenv (the same one Nemotron's own server uses). Without this, the script silently
# failed with "vllm: command not found" and the process exited immediately, which looked
# like Cosmos was running (backgrounded, no error surfaced) when it never actually started.
source /opt/dlami/nvme/physyk/workspace/.venv-nemotron/bin/activate
vllm serve nvidia/Cosmos-Reason2-2B \
  --served-model-name nvidia/Cosmos-Reason2-2B \
  --port 8001 --host 0.0.0.0 \
  --gpu-memory-utilization 0.14 \
  --max-model-len 8192 \
  --limit-mm-per-prompt '{"image": 2, "video": 0}' \
  --reasoning-parser qwen3 \
  --allowed-local-media-path / \
  --no-enable-log-requests
