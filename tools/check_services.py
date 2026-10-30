"""Connectivity + format check for the two model servers, with no sim and no GUI. Run it from any python 3:
    python tools/check_services.py            (reads .env via workspace/physyk_config.py)
It lists the served models, asks Nemotron for a tiny JSON plan (the orchestrator's parser is applied to the reply) and sends
Cosmos one saved overview frame. Read-only: nothing is started or changed."""
import base64, json, sys, time, urllib.request
from pathlib import Path

WS = Path(__file__).resolve().parent.parent / "workspace"
sys.path.insert(0, str(WS))
import physyk_config as cfg


def call(url, payload=None, key=""):
    req = urllib.request.Request(url, data=json.dumps(payload).encode() if payload else None,
                                 headers=cfg.auth_headers(key), method="POST" if payload else "GET")
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=90) as r:
        return json.loads(r.read()), time.time() - t0


def main():
    for name, base, key in (("Nemotron", cfg.NEMOTRON_BASE_URL, cfg.NEMOTRON_API_KEY), ("Cosmos", cfg.COSMOS_BASE_URL, cfg.COSMOS_API_KEY)):
        try:
            body, dt = call(f"{base}/models", key=key)
            print(f"[{name}] {base}/models OK {dt:.1f}s -> {[m.get('id') for m in body.get('data', [])]}")
        except Exception as e:
            print(f"[{name}] {base}/models FAILED: {e}")
    print(f"configured: NEMOTRON_MODEL={cfg.NEMOTRON_MODEL!r} COSMOS_MODEL={cfg.COSMOS_MODEL!r}")

    # Nemotron: one-step plan, parsed with the orchestrator's own parser.
    import physyk_agent_orchestrator as orch
    scene = {"objects": {"red_cube": {"name": "red cube", "position": [0.4, 0.1, 0.025]}, "tray": {"name": "tray", "position": [0.42, -0.32, 0.015]}}}
    try:
        t0 = time.time()
        raw = orch._call_nemotron("put the red cube in the tray", scene)
        print(f"[Nemotron] reply in {time.time() - t0:.1f}s, {len(raw)} chars; has </think>: {'</think>' in raw}")
        print("[Nemotron] parsed:", orch._parse_subgoals(raw))
    except Exception as e:
        print(f"[Nemotron] plan test FAILED: {e}")

    # Cosmos: the saved overview frame, same call the step verifier uses.
    frame = WS / "perception_calib" / "frame.jpg"
    if frame.is_file():
        from physyk_perception import cosmos_client
        res, ms, err = cosmos_client.ask_visual_question(
            frame.read_bytes(), "image/jpeg",
            'How many small cubes are on the table? Answer with ONLY JSON: {"count": <int>, "observed": "<one sentence>"}',
            timeout=cfg.COSMOS_TIMEOUT_S)
        print(f"[Cosmos] {ms:.0f} ms err={err} result={res}")
    else:
        print("[Cosmos] no perception_calib/frame.jpg, skipped")


if __name__ == "__main__":
    main()
