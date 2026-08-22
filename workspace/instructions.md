# 🚀 Physyk — Run Guide

## Access

1. Windows VM → forward port `3000` (VS Code Brev access).
2. Open `https://vscode-rrx2yrnxm.apps.run.brev.nvidia.com/?folder=/opt/dlami`
3. Forward the app ports (local machine):
   ```bash
   brev port-forward rrx2yrnxm -p 7860:7860 -p 8100:8100 -p 8000:8000 -p 8300:8300 -p 8210:8210
   ```
4. Start the stack (in the Brev terminal):
   ```bash
   cd /opt/dlami/nvme/physyk/workspace
   ./run.sh --all
   bash start_groot.sh &   # only if you need GR00T VLA (`vla:` commands)
   ```
5. Open the GUI at the URL Brev lists for **7860**.

## Ports

| Port | Service | Started by |
|---|---|---|
| **7860** | Web GUI — use this one | `run.sh` |
| **8100** | Isaac Sim (physics/camera) — proxied through 7860, rarely needed directly | `run.sh` |
| **8000** | Nemotron-30B | `run.sh --all` |
| **8300** | GR00T-N1.7 VLA server | `start_groot.sh` (manual, separate) |
| **8210** | Isaac Sim WebRTC | *nothing starts this currently* — legacy, safe to skip |
| **3000** | VS Code Brev access | Brev (not Physyk) |

## Commands

```bash
./run.sh              # Isaac Sim + GUI
./run.sh --all         # + Nemotron
./run.sh --status      # health check
./run.sh --stop        # stop (does NOT stop GR00T server)
bash start_groot.sh    # GR00T VLA server, separately
pkill -f groot_policy_service.py   # stop it
bash restart_physyk.sh # fallback: restart Isaac Sim + GUI, waits for health
```

## Gotcha

Isaac Sim's own internal redirect link sometimes fails through Brev. Ignore it — open the URL
**Brev itself lists for port 7860** instead; the GUI already proxies the camera feed.

## Health & logs

```bash
curl -s localhost:7860/health; curl -s localhost:8100/health
curl -s localhost:8000/health; curl -s localhost:8300/health   # if running
tail -f /tmp/isaac_sim_service.log /tmp/physyk_server.log
tail -f /opt/dlami/nvme/physyk/logs/{nemotron,groot}_server.log   # if running
```

---
Architecture / what's done vs. in progress → `PLAN.md`.
