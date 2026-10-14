# Arm test tools

Scripts used to measure the Franka arm's physics and the pick-and-place behaviour (results: `TECHNICAL_DEEP_DIVE.md` §1.3, §1.8, §5).
Run everything from the repo root.

## End-to-end, on the running sim (port 8100)

These talk to the **single sim instance** over HTTP. Start it first (`./start_sim_docker.sh`); to compare settings, restart that same sim with
the setting in front of the command and a fixed layout seed:

```bash
PHYSYK_ARM_OBSTACLES=off PHYSYK_RANDOM_SEED=7 ./start_sim_docker.sh      # in another terminal
.venv/bin/python workspace/scripts/arm_tests/e2e_batch.py off 6 /tmp/e2e_off.json
.venv/bin/python workspace/scripts/arm_tests/analyze_e2e.py /tmp/e2e_off.json
```

`e2e_batch.py` resets, randomizes, plans a stack with the agent's own planner (tray on even layouts, table on odd ones) and sends each pick through
`/execute`; `analyze_e2e.py` reads `workspace/joint_trace.csv` (written by the sim while it ran) and prints per-pick peak joint speed vs the
Franka caps, frames at the cap, acceleration vs the rated limits, joint path and "swivel" frames. A batch of 6 layouts takes about 7 minutes.

## Standalone physics harness (its own throw-away container, no port)

These reproduce a pick-and-place outside the sim service so single settings can be varied. They start a **separate** Isaac Sim container, so
use them only when the user agrees (they share the GPU with the sim).

```bash
docker run --rm --name arm-test --gpus all --network host \
  -e ACCEPT_EULA=Y -e PRIVACY_CONSENT=Y -e OMNI_KIT_ACCEPT_EULA=YES \
  -v "$PWD/.isaac-cache/kit:/isaac-sim/kit/cache:rw" -v "$PWD/.isaac-cache/ov:/root/.cache/ov:rw" \
  -v "$PWD/.isaac-cache/glcache:/root/.cache/nvidia/GLCache:rw" -v "$PWD/.isaac-cache/computecache:/root/.nv/ComputeCache:rw" \
  -v "$PWD/workspace:/workspace:ro" -v "$PWD/logs:/out:rw" -w /workspace \
  --entrypoint /isaac-sim/python.sh nvcr.io/nvidia/isaac-sim:6.0.1 scripts/arm_tests/<script>.py > /tmp/<script>.out 2>&1
```

| Script | What it does |
|---|---|
| `dump_arm.py` | Prints the arm's drive gains, effort / velocity limits, joint limits, armature, friction, solver iterations and link masses as the sim loads them |
| `arm_trace.py` | One instrumented pick-and-place per `run(...)` line: every step's controller phase, joints, RMPFlow's commanded positions and velocities, hand and target positions → `logs/trace_<tag>.csv` |
| `arm_sweep.py` | The same move under different speed / damping / stiffness / arrival-gating / obstacle / hand-yaw / RMPFlow-parameter settings; one `RESULT` json line per run |

Notes: the container user cannot read files in private temp directories, so the scripts are run from `workspace/` as above. The standalone harness
registers obstacles only when asked, and **after** the controller reset (`PickPlaceController.reset()` clears registered obstacles). The scripts were
compile-checked after being moved here; their logic is unchanged from the versions that produced the documented results, except that the
experimental joint-speed clip option of the sweep was removed (it never worked).
