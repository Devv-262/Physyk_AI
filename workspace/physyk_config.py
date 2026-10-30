"""Single source of truth for where each service lives and how to authenticate to it.

Everything runs on this laptop except the Nemotron planner, which defaults to NVIDIA's hosted API
(https://integrate.api.nvidia.com/v1); Cosmos 3 Nano Reasoner runs locally as a NIM container (port 8001). Override any value with an env var, or put them in a
`.env` file next to this module / in the project root (see .env.example).

  NVIDIA_API_KEY        nvapi-... key from build.nvidia.com (used for both models unless overridden)
  NEMOTRON_BASE_URL     default https://integrate.api.nvidia.com/v1  (a local vLLM: http://localhost:8000/v1)
  NEMOTRON_MODEL        model id as listed on build.nvidia.com
  NEMOTRON_API_KEY      overrides NVIDIA_API_KEY for Nemotron only
  COSMOS_BASE_URL       default http://localhost:8001/v1  (the local Cosmos 3 Nano Reasoner NIM container)
  COSMOS_MODEL          default nvidia/cosmos3-nano-reasoner
  COSMOS_API_KEY_FOR_REQUESTS  bearer token sent to Cosmos (none needed for the local NIM)
  ISAAC_SIM_URL         default http://localhost:8100
"""

import os
from pathlib import Path


def _load_dotenv():
    here = Path(__file__).resolve().parent
    for path in (here / ".env", here.parent / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"

# Ports: defaults are the Linux/Docker ones; override them in .env if needed.
PHYSYK_GUI_PORT = int(os.environ.get("PHYSYK_GUI_PORT", "7860"))
PHYSYK_SIM_PORT = int(os.environ.get("PHYSYK_SIM_PORT", "8100"))
ISAAC_SIM_URL = os.environ.get("ISAAC_SIM_URL", f"http://localhost:{PHYSYK_SIM_PORT}")

NEMOTRON_BASE_URL = os.environ.get("NEMOTRON_BASE_URL", NVIDIA_API_BASE).rstrip("/")
NEMOTRON_MODEL = os.environ.get("NEMOTRON_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
NEMOTRON_API_KEY = os.environ.get("NEMOTRON_API_KEY") or os.environ.get("NVIDIA_API_KEY", "")

# Cosmos 3 Nano Reasoner is NOT on the hosted API — it's self-hosted as a NIM container on this
# laptop (./run.sh --cosmos, port 8001). No bearer token is sent to it: COSMOS_API_KEY in
# .env is only the NGC key the container uses to download weights.
COSMOS_BASE_URL = os.environ.get("COSMOS_BASE_URL", "http://localhost:8001/v1").rstrip("/")
COSMOS_MODEL = os.environ.get("COSMOS_MODEL", "nvidia/cosmos3-nano-reasoner")
COSMOS_API_KEY = os.environ.get("COSMOS_API_KEY_FOR_REQUESTS", "")

# Timeouts / budgets (all optional). A Cosmos on another machine needs more than the local 6 s.
NEMOTRON_TIMEOUT_S = float(os.environ.get("NEMOTRON_TIMEOUT_S", "90"))
COSMOS_TIMEOUT_S = float(os.environ.get("COSMOS_TIMEOUT_S", "6"))
COSMOS_VERIFY_TIMEOUT_S = float(os.environ.get("COSMOS_VERIFY_TIMEOUT_S", "6"))
COSMOS_MAX_TOKENS = int(os.environ.get("COSMOS_MAX_TOKENS", "512"))


def auth_headers(api_key: str) -> dict:
    """Content-Type plus a Bearer token when a key is configured (a local vLLM needs none)."""
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers
