#!/usr/bin/env python3
"""
Cosmos-Reason2-2B smoke test — standalone, no Isaac Sim integration.

Builds a synthetic image (colored squares on a plain background, roughly resembling the
real red/blue/green cube + tray scene) and sends it to the served Cosmos-Reason2-2B model
(port 8001, OpenAI-compatible /v1/chat/completions) with a real grounding-style prompt.
Prints latency and the parsed response. Proves the server/model itself works in isolation,
same discipline used for the Nemotron server earlier this session — isolate server/prompt
problems from sim/camera problems before touching Isaac Sim at all.
"""

import base64
import io
import json
import re
import time
import urllib.request

from PIL import Image, ImageDraw

COSMOS_URL = "http://localhost:8001/v1/chat/completions"


def build_synthetic_scene() -> str:
    """A plain gray tabletop with three colored squares (cubes) and a yellow tray
    rectangle — same rough composition as the real overview camera frame."""
    img = Image.new("RGB", (1280, 720), (120, 150, 170))
    draw = ImageDraw.Draw(img)
    # Tray (yellow rectangle, lower-left-ish)
    draw.rectangle([180, 480, 420, 620], fill=(230, 210, 60))
    # Cubes
    draw.rectangle([550, 380, 640, 470], fill=(210, 60, 60))    # red cube
    draw.rectangle([700, 350, 790, 440], fill=(60, 90, 210))    # blue cube
    draw.rectangle([850, 400, 940, 490], fill=(60, 190, 90))    # green cube
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def main():
    image_b64 = build_synthetic_scene()
    payload = {
        "model": "nvidia/Cosmos-Reason2-2B",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        "This image shows a tabletop robotic workspace with a red cube, "
                        "a blue cube, a green cube, and a yellow tray. For the red cube, "
                        "report: is it visible (yes/no), and its approximate center point "
                        "as [x, y] pixel coordinates. Respond with ONLY a JSON object: "
                        '{"visible": true/false, "point": [x, y]}'
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                ],
            }
        ],
        "temperature": 0.0,
        "max_tokens": 512,
    }
    req = urllib.request.Request(
        COSMOS_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=60.0) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    latency_ms = (time.time() - t0) * 1000.0

    msg = body["choices"][0]["message"]
    # Confirmed by direct testing (not assumed): this reasoning-parser (--reasoning-parser
    # qwen3) setup puts the model's actual answer in `reasoning`, not the usual `content`
    # field, whenever the answer stays inside its chain-of-thought without a separate final
    # message ever being emitted.
    raw_answer = msg.get("content") or msg.get("reasoning")
    print(f"Latency: {latency_ms:.1f} ms")
    print(f"Raw response ({'content' if msg.get('content') else 'reasoning'} field):\n{raw_answer}")

    real_center = [(550 + 640) // 2, (380 + 470) // 2]
    print(f"\nKnown real red-cube pixel center in the synthetic image: {real_center}")

    match = re.search(r"\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?", raw_answer or "")
    if match:
        px, py = float(match.group(1)), float(match.group(2))
        # Confirmed by direct testing: this model's grounding coordinates are normalized
        # 0-1000 relative to image dimensions, not raw pixels (a magenta square at true
        # pixel center [1050, 150] in a 1280x720 image was reported as [811, 196] — wildly
        # wrong as raw pixels, but rescaling as norm1000 -> [1038, 141], within ~12px of
        # truth). Applying the same rescale here.
        rescaled = [px / 1000.0 * 1280.0, py / 1000.0 * 720.0]
        err = ((rescaled[0] - real_center[0]) ** 2 + (rescaled[1] - real_center[1]) ** 2) ** 0.5
        print(f"Parsed model point (raw, presumed norm1000): [{px:.0f}, {py:.0f}]")
        print(f"Rescaled to pixels: [{rescaled[0]:.0f}, {rescaled[1]:.0f}]")
        print(f"Error vs real center: {err:.1f} px")
    else:
        print("Could not parse a [x, y] point out of the response.")


if __name__ == "__main__":
    main()
