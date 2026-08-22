"""Grounding calls to the served Cosmos-Reason2-2B model (port 8001, OpenAI-compatible
/v1/chat/completions). Reuses exactly the parsing behavior confirmed empirically earlier in
this project (cosmos_smoke_test.py, perception_calib/): the real answer often lands in the
`reasoning` field (not `content`) because the server runs with --reasoning-parser qwen3, and
grounding coordinates are normalized 0-1000 relative to image dimensions ("norm1000"), not
raw pixels — confirmed twice, once on a synthetic image and once on a real captured Isaac Sim
frame, with real pixel-error measurements both times.
"""

import base64
import io
import json
import re
import time
import urllib.request
import urllib.error

import numpy as np
from PIL import Image

COSMOS_URL = "http://localhost:8001/v1/chat/completions"
DEFAULT_TIMEOUT_S = 6.0  # per cosmos_integration.md's own guard rail for shadow/vision mode


def _encode_rgb_png(rgb: np.ndarray) -> str:
    img = Image.fromarray(rgb[:, :, :3].astype(np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _post_chat(image_b64: str, mime: str, prompt: str, timeout: float):
    """Shared low-level call: posts one image + text prompt to Cosmos, returns
    (raw_answer_text, latency_ms, error). Never raises — every failure path (timeout,
    connection refused, malformed response) comes back as a non-None error string instead,
    so callers can degrade gracefully rather than needing their own try/except around this."""
    payload = {
        "model": "nvidia/Cosmos-Reason2-2B",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
            ],
        }],
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return None, (time.time() - t0) * 1000.0, f"request failed: {e}"
    except Exception as e:
        return None, (time.time() - t0) * 1000.0, f"unexpected error: {e}"
    latency_ms = (time.time() - t0) * 1000.0

    try:
        msg = body["choices"][0]["message"]
        raw_answer = msg.get("content") or msg.get("reasoning") or ""
    except (KeyError, IndexError, TypeError) as e:
        return None, latency_ms, f"malformed response: {e}"
    return raw_answer, latency_ms, None


def ask_visual_question(image_bytes: bytes, mime: str, question: str, timeout: float = DEFAULT_TIMEOUT_S):
    """General-purpose visual Q&A call (used by step_verifier.py's post-condition checks,
    Step 7) — takes raw already-encoded image bytes (e.g. a JPEG straight off Isaac Sim's
    /camera/scene.jpg, no decode/re-encode needed) rather than an ndarray, so callers that
    only have HTTP-fetched image bytes don't need numpy/PIL themselves.

    Returns (parsed_json_dict, latency_ms, error) — error is None only on a fully successful
    round trip with a parseable JSON object in the response; callers must treat any non-None
    error as "no usable answer this call", never raise."""
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    raw_answer, latency_ms, err = _post_chat(image_b64, mime, question, timeout)
    if err is not None:
        return None, latency_ms, err
    match = re.search(r"\{.*\}", raw_answer, re.DOTALL)
    if not match:
        return None, latency_ms, "no JSON object found in response"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return None, latency_ms, f"JSON parse failed: {e}"
    return parsed, latency_ms, None


def ground_objects(rgb: np.ndarray, labels: list, timeout: float = DEFAULT_TIMEOUT_S):
    """Sends ONE grounding call covering ALL requested labels (cheaper than one call per
    object, and per cosmos_integration.md's own advice for wiring a whole plan step at once).

    Returns (points, latency_ms, error) where points is a dict {label: (u, v) pixel in THIS
    image's real pixel space, already rescaled from norm1000} or {label: None} for anything
    Cosmos reports as not visible / fails to parse. error is None on success, else a short
    string describing what went wrong (timeout, HTTP error, bad JSON, ...) — callers must
    treat any non-None error as "no usable perception this call", not raise.
    """
    height, width = rgb.shape[0], rgb.shape[1]
    image_b64 = _encode_rgb_png(rgb)
    label_list = ", ".join(labels)
    prompt = (
        f"This image shows a tabletop robotic workspace. Locate each of these objects: "
        f"{label_list}. For each one, report whether it is visible and its approximate "
        f"center point as [x, y] coordinates normalized to a 0-1000 range (0,0 is the "
        f"top-left corner, 1000,1000 is the bottom-right corner). Respond with ONLY a JSON "
        f"object mapping each object name to either its [x, y] point, or null if not "
        f"visible. Example: " + json.dumps({labels[0]: [500, 500]}) if labels else "{}"
    )
    payload = {
        "model": "nvidia/Cosmos-Reason2-2B",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            ],
        }],
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
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {label: None for label in labels}, (time.time() - t0) * 1000.0, f"request failed: {e}"
    except Exception as e:
        return {label: None for label in labels}, (time.time() - t0) * 1000.0, f"unexpected error: {e}"
    latency_ms = (time.time() - t0) * 1000.0

    try:
        msg = body["choices"][0]["message"]
        raw_answer = msg.get("content") or msg.get("reasoning") or ""
    except (KeyError, IndexError, TypeError) as e:
        return {label: None for label in labels}, latency_ms, f"malformed response: {e}"

    # Model sometimes wraps the JSON in prose or a code fence — pull out the first
    # balanced-looking {...} block rather than assuming the whole string is clean JSON.
    match = re.search(r"\{.*\}", raw_answer, re.DOTALL)
    if not match:
        return {label: None for label in labels}, latency_ms, "no JSON object found in response"
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as e:
        return {label: None for label in labels}, latency_ms, f"JSON parse failed: {e}"

    points = {}
    for label in labels:
        val = parsed.get(label)
        if not val or not isinstance(val, (list, tuple)) or len(val) < 2:
            points[label] = None
            continue
        try:
            nx, ny = float(val[0]), float(val[1])
        except (TypeError, ValueError):
            points[label] = None
            continue
        # norm1000 -> real pixel space of THIS image (confirmed convention, see module docstring)
        px = nx / 1000.0 * width
        py = ny / 1000.0 * height
        points[label] = (px, py)
    return points, latency_ms, None
