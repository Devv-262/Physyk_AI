"""Step-at-a-time Nemotron planner: one short answer per turn, streamed, with a standing system prompt.

For prompts that name colours or sequence steps ("put red on blue, then blue in the tray, ..."), planning the whole
sequence up front is slow and brittle: the model has to simulate every move, including where the blocking cubes
should be parked. Instead the agent asks for ONE next move, executes it, shows the model the verified result and the
live scene, and asks for the next one, until the model says the goal holds.

Why it is cheap:
  * the system prompt is static (no scene in it), so every request starts with the same prefix and the hosted API's
    prefix cache serves it (usage.prompt_tokens_details.cached_tokens); `warm_up()` primes it at start-up;
  * thinking is switched off (chat_template_kwargs.enable_thinking=false): the answer carries a one-sentence "say" instead,
    which is the only reasoning the move needs once the state is spelled out;
  * the scene is described by code (what is on what, what is covered), so the model does not derive it from coordinates;
  * history is rebuilt each turn: the current goal's moves and results, older goals as one-line summaries, and only the
    LATEST scene (older scenes are stale and would only cost tokens).

The API is stateless; "keeping" the prompt means keeping this prefix identical, not a server-side session.
"""

import json
import math
import os
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import httpx

import physyk_config as cfg

# Measured settings (see TECHNICAL_DEEP_DIVE.md): thinking off, temperature 0, a small completion cap. With thinking on the same
# answer took 5-10x the tokens; the cap is generous for a one-object answer (typically 40-70 tokens).
PLAN_MAX_TOKENS = int(os.environ.get("NEMOTRON_STEP_MAX_TOKENS", "400"))
PLAN_TEMPERATURE = float(os.environ.get("NEMOTRON_STEP_TEMPERATURE", "0"))
# "off" (default): no chain-of-thought. "on": thinking with a hard token ceiling (THINK_MAX_TOKENS covers thought + answer).
PLAN_THINKING = os.environ.get("NEMOTRON_STEP_THINKING", "off").lower()
THINK_MAX_TOKENS = int(os.environ.get("NEMOTRON_THINK_MAX_TOKENS", "2500"))
STEP_TIMEOUT_S = 45.0
MAX_TURNS_PER_GOAL = 14           # hard stop on a runaway goal (a 4-move request needs about 6)

STEP_SYSTEM_PROMPT = """You plan moves for a Franka Panda arm on a work table. Objects: a red cube, a green cube, a blue cube \
(5 cm each) and a yellow tray. "destination" is always one of: "in the tray", "on top of the <colour> cube", \
"on a free spot on the table", or "right of / left of / in front of / behind / next to the <colour> cube".

You are called in two ways. Answer with exactly one JSON object and nothing else.

A) The user gives a GOAL and no MILESTONES yet. Split it into its ordered FINAL placements, in the user's order, one per clause:
  {"say": "<max 20 words>", "milestones": [{"target": "<colour> cube", "destination": "<place>"}, ...]}
  or {"say": "<why>", "impossible": "<reason>"} if it names something that does not exist or contradicts itself.
  Do NOT add parking moves here; only what the user asked for. "A over B", "A on B", "A onto B" = A goes on top of B. "in/into the tray" = in \
the tray. "on the table" = "on a free spot on the table".
  Example: GOAL "put red on blue, then blue in the tray" ->
  {"say": "Two placements in order.", "milestones": [{"target": "red cube", "destination": "on top of the blue cube"}, {"target": "blue cube", "destination": "in the tray"}]}

B) You are shown MILESTONES with [done] / [current] / [later] marks and the live STATE. Give the ONE next move that advances the [current] milestone:
  {"say": "<max 25 words: what is in the way, if anything>", "next": {"action": "pick_place", "target": "<colour> cube", "destination": "<place>"}}
  {"say": "<why>", "impossible": "<reason>"}
Rules for B:
1. If the [current] milestone's cube is free to pick AND (for "on top of X") X is free on top, make exactly the milestone's move.
2. A COVERED cube cannot be picked. If the cube to move is covered, move the cube on top of it first; if the base X you must place on is covered, \
move what covers X first. Park such a cube with "on a free spot on the table". Never target a covered cube.
3. Never put a cube on top of itself or on top of a cube that already has one on it.
4. Moving a cube takes it alone: whatever was on it must be moved off first, whatever it sat on stays.
5. Work only on the [current] milestone. Ignore [done] ones (they were satisfied at their time and may have been taken apart on purpose) and do \
not start [later] ones. Do not repeat a move that STATE already shows.
6. If the last move FAILED, read the new STATE: retry once, otherwise pick another way, otherwise impossible.
7. Only objects listed in STATE exist. NEVER substitute another object for one that is missing or that the robot refused: answer impossible \
with the reason instead.
"""

_JSON_OBJ_START = "{"


def _name(scene: Dict[str, Any], key: str) -> str:
    return str((scene.get("objects", {}) or {}).get(key, {}).get("name", key)).lower()


def describe_state(scene: Dict[str, Any], tray_xy=(0.42, -0.32), tray_radius=0.13,
                   stack_xy=0.035, stack_dz=0.035) -> str:
    """The scene as relations, not coordinates: where each cube is and what covers it. Computed here so the model does not
    have to derive stacking from numbers."""
    objs = scene.get("objects", {}) or {}
    cubes = {k: v for k, v in objs.items() if k.endswith("_cube") and v.get("position") is not None}
    if not cubes:
        return "(no cubes visible)"
    on_top: Dict[str, Optional[str]] = {k: None for k in cubes}     # cube -> the cube resting directly on it
    below: Dict[str, Optional[str]] = {k: None for k in cubes}      # cube -> the cube it rests on
    for k, v in cubes.items():
        p = v["position"]
        best, best_z = None, None
        for o, w in cubes.items():
            if o == k:
                continue
            q = w["position"]
            if math.hypot(q[0] - p[0], q[1] - p[1]) < stack_xy and q[2] - p[2] > stack_dz:
                if best is None or q[2] < best_z:
                    best, best_z = o, q[2]
        on_top[k] = best
        if best is not None:
            below[best] = k
    lines = []
    for k, v in sorted(cubes.items(), key=lambda kv: _name(scene, kv[0])):
        p = v["position"]
        if below[k] is not None:
            where = f"on top of the {_name(scene, below[k])}"
        elif math.hypot(p[0] - tray_xy[0], p[1] - tray_xy[1]) < tray_radius:
            where = "in the tray"
        else:
            where = "on the table"
        cover = f", COVERED by the {_name(scene, on_top[k])}" if on_top[k] else ", free to pick"
        lines.append(f"- {_name(scene, k)}: {where}{cover}")
    missing = [n for n in ("red", "green", "blue") if f"{n}_cube" not in cubes]
    if missing:
        lines.append("- not visible: " + ", ".join(f"{m} cube" for m in missing))
    return "\n".join(lines)


def cube_locations(scene: Dict[str, Any], tray_xy=(0.42, -0.32), tray_radius=0.13, stack_xy=0.035, stack_dz=0.035):
    """{cube key: ("on", base key) | ("tray",) | ("table",)} and {cube key: covering cube key or None}, from positions."""
    objs = scene.get("objects", {}) or {}
    cubes = {k: v["position"] for k, v in objs.items() if k.endswith("_cube") and v.get("position") is not None}
    covered_by: Dict[str, Optional[str]] = {k: None for k in cubes}
    base_of: Dict[str, Optional[str]] = {k: None for k in cubes}
    for k, p in cubes.items():
        best, best_z = None, None
        for o, q in cubes.items():
            if o != k and math.hypot(q[0] - p[0], q[1] - p[1]) < stack_xy and q[2] - p[2] > stack_dz:
                if best is None or q[2] < best_z:
                    best, best_z = o, q[2]
        covered_by[k] = best
        if best is not None:
            base_of[best] = k
    loc = {}
    for k, p in cubes.items():
        if base_of[k] is not None:
            loc[k] = ("on", base_of[k])
        elif math.hypot(p[0] - tray_xy[0], p[1] - tray_xy[1]) < tray_radius:
            loc[k] = ("tray",)
        else:
            loc[k] = ("table",)
    return loc, covered_by


def _cube_key(name: str) -> Optional[str]:
    n = (name or "").lower()
    for c in ("red", "green", "blue"):
        if c in n:
            return f"{c}_cube"
    return None


def milestone_holds(milestone: Dict[str, Any], scene: Dict[str, Any]) -> Optional[bool]:
    """Does the milestone's placement hold in the scene right now? True / False, or None when it cannot be judged from
    positions (relative placements such as "right of"): the caller then trusts the verified move."""
    loc, _ = cube_locations(scene)
    k = _cube_key(milestone.get("target", ""))
    if k is None or k not in loc:
        return False
    d = (milestone.get("destination") or "").lower()
    if "top of" in d:
        base = _cube_key(d)
        return base is not None and loc[k] == ("on", base)
    if "tray" in d:
        return loc[k] == ("tray",)
    if "table" in d or "free spot" in d:
        return loc[k] == ("table",)
    return None


def advance(milestones: List[Dict[str, Any]], current: int, scene: Dict[str, Any],
            last_move: Optional[Dict[str, Any]] = None, last_ok: bool = False) -> int:
    """The index of the first milestone, from `current` on, that does not hold yet (len(milestones) = all reached). A milestone
    counts as reached when the scene shows it, or - for placements positions cannot judge - when the verified move was it."""
    while current < len(milestones):
        m = milestones[current]
        holds = milestone_holds(m, scene)
        same_move = (last_move is not None and last_ok and _cube_key(last_move.get("target", "")) == _cube_key(m.get("target", ""))
                     and (last_move.get("destination") or "").lower().strip() == (m.get("destination") or "").lower().strip())
        if holds is True or (holds is None and same_move):
            current += 1
            last_move = None
            continue
        return current
    return current


class StepPlanner:
    """One instance serves the agent's lifetime and many prompts (older goals stay as one-line summaries). The goal is split
    into milestones once (`plan_goals`); then `next_move` is asked for one move toward the current milestone, and CODE decides
    when a milestone is reached (`milestone_holds`, or the verified move itself), so the model never has to remember progress."""

    def __init__(self, on_stream: Optional[Callable[[str, str], None]] = None):
        self._lock = threading.Lock()
        self.goal: Optional[str] = None
        self.milestones: List[Dict[str, Any]] = []
        self.moves: List[Tuple[Dict[str, Any], str]] = []     # (the model's full answer {say, next}, the verified result text)
        self.history: List[str] = []                          # one line per finished goal
        self.on_stream = on_stream
        self.last_stats: Dict[str, Any] = {}

    def start_goal(self, goal: str) -> None:
        with self._lock:
            if self.goal:
                self.history.append(f'"{self.goal}" -> {len(self.moves)} move(s)')
                self.history = self.history[-6:]
            self.goal, self.milestones, self.moves = goal, [], []

    def record_result(self, answer: Dict[str, Any], result: str) -> None:
        with self._lock:
            self.moves.append((answer, result))

    def _preamble(self) -> List[Dict[str, str]]:
        msgs: List[Dict[str, str]] = [{"role": "system", "content": STEP_SYSTEM_PROMPT}]
        if self.history:
            msgs.append({"role": "user", "content": "Earlier goals (finished, context only): " + "; ".join(self.history)})
            msgs.append({"role": "assistant", "content": '{"say": "Noted.", "impossible": "none"}'})
        return msgs

    def _call(self, messages: List[Dict[str, str]], expect: str) -> Dict[str, Any]:
        """`expect`: the key the answer must carry (\"milestones\" for call A, \"next\" for call B), else it is asked again once."""
        payload: Dict[str, Any] = {
            "model": cfg.NEMOTRON_MODEL, "messages": messages, "temperature": PLAN_TEMPERATURE, "stream": True,
            "stream_options": {"include_usage": True},
        }
        if PLAN_THINKING == "on":
            payload["max_tokens"] = THINK_MAX_TOKENS
        else:
            payload["max_tokens"] = PLAN_MAX_TOKENS
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        content, reasoning, usage, t_first, t_total = _stream_chat(payload, self.on_stream)
        try:
            answer = _parse_answer(content)
            if expect not in answer and "impossible" not in answer and "done" not in answer:
                raise RuntimeError("answer has the wrong shape")
        except RuntimeError:
            # one retry with a nudge: an occasional answer comes back as prose or an unknown shape
            payload["messages"] = messages + [{"role": "assistant", "content": content or "..."},
                                              {"role": "user", "content": f'Reply with the single JSON object only, in answer shape {"A" if expect == "milestones" else "B"} (key "{expect}").'}]
            content, reasoning, usage, t_first, t_total = _stream_chat(payload, self.on_stream)
            answer = _parse_answer(content)
            if expect not in answer and "impossible" not in answer and "done" not in answer:
                raise RuntimeError(f"Nemotron kept answering in the wrong shape: {content[:160]!r}")
        answer["reasoning"] = reasoning
        answer["stats"] = {"first_token_s": t_first, "total_s": t_total, "usage": usage}
        self.last_stats = answer["stats"]
        return answer

    def plan_goals(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """Call A: split the goal into ordered milestones. Returns {"milestones": [...]} or {"impossible": ...}."""
        with self._lock:
            msgs = self._preamble() + [{"role": "user", "content": f'ANSWER SHAPE A. GOAL: "{self.goal}"\nSTATE:\n{describe_state(scene)}'}]
        ans = self._call(msgs, "milestones")
        if "milestones" in ans:
            with self._lock:
                self.milestones = ans["milestones"]
        return ans

    def next_move(self, scene: Dict[str, Any], current: int) -> Dict[str, Any]:
        """Call B: one move toward milestone `current` (0-based). Returns {"next": ...} | {"impossible": ...}."""
        with self._lock:
            if len(self.moves) >= MAX_TURNS_PER_GOAL:
                return {"impossible": f"gave up after {MAX_TURNS_PER_GOAL} moves", "say": "move limit reached", "stats": {}}
            marks = []
            for i, m in enumerate(self.milestones):
                tag = "done" if i < current else ("current" if i == current else "later")
                marks.append(f'  {i + 1}. [{tag}] {m.get("target")} -> {m.get("destination")}')
            msgs = self._preamble() + [{"role": "user", "content": f'GOAL: "{self.goal}"'}]
            for answer, result in self.moves[-4:]:        # the last few moves are enough context; older ones only cost tokens
                msgs.append({"role": "assistant", "content": json.dumps(answer, separators=(", ", ": "))})
                msgs.append({"role": "user", "content": f"RESULT: {result}"})
            if msgs[-1]["role"] == "user":                # merge into one user message (alternating roles)
                msgs[-1]["content"] += "\n"
            else:
                msgs.append({"role": "user", "content": ""})
            msgs[-1]["content"] += "ANSWER SHAPE B. MILESTONES:\n" + "\n".join(marks) + f"\nSTATE:\n{describe_state(scene)}"
        return self._call(msgs, "next")


def _stream_chat(payload: Dict[str, Any], on_stream=None):
    """Server-sent-event stream (tokens are handed to `on_stream` as they arrive). Returns (content, reasoning, usage, first_token_s, total_s)."""
    url = f"{cfg.NEMOTRON_BASE_URL}/chat/completions"
    headers = cfg.auth_headers(cfg.NEMOTRON_API_KEY)
    content, reasoning, usage = "", "", None
    t0 = time.time()
    t_first = None
    last_err = None
    for attempt in range(4):
        content, reasoning, usage, t_first = "", "", None, None
        try:
            with httpx.stream("POST", url, json=payload, headers=headers, timeout=STEP_TIMEOUT_S) as r:
                if r.status_code in (429, 500, 502, 503, 504):
                    last_err = f"HTTP {r.status_code}"
                    try:
                        wait = float(r.headers.get("retry-after", ""))
                    except ValueError:
                        wait = 2.0 ** (attempt + 1)               # 2, 4, 8, 16 s: the hosted tier rate-limits bursts
                    time.sleep(min(wait, 20.0))
                    continue
                if r.status_code != 200:
                    body = r.read().decode("utf-8", "replace")[:300]
                    raise RuntimeError(f"Nemotron HTTP {r.status_code}: {body}")
                for line in r.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    if chunk.get("usage"):
                        usage = chunk["usage"]
                    for ch in chunk.get("choices") or []:
                        delta = ch.get("delta") or {}
                        r_txt = delta.get("reasoning_content") or delta.get("reasoning") or ""
                        c_txt = delta.get("content") or ""
                        if (r_txt or c_txt) and t_first is None:
                            t_first = round(time.time() - t0, 2)
                        if r_txt:
                            reasoning += r_txt
                            if on_stream:
                                on_stream("reasoning", r_txt)
                        if c_txt:
                            content += c_txt
                            if on_stream:
                                on_stream("content", c_txt)
                    if "</think>" in content:
                        reasoning, _, content = content.partition("</think>")
            return content, reasoning, usage, t_first, round(time.time() - t0, 2)
        except (httpx.HTTPError, OSError) as e:
            last_err = str(e)
            time.sleep(1.0 * (attempt + 1))
    raise RuntimeError(f"Nemotron request failed: {last_err}")


def _first_json_object(text: str) -> Optional[str]:
    """The first balanced {...} in text (string-literal aware), or None while it is still incomplete."""
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth > 0:
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def _parse_answer(content: str) -> Dict[str, Any]:
    obj = _first_json_object(content)
    if obj is None:
        raise RuntimeError(f"no JSON object in Nemotron answer: {content[:200]!r}")
    try:
        data = json.loads(obj)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"bad JSON from Nemotron: {e}: {obj[:200]!r}")
    if not isinstance(data, dict):
        raise RuntimeError(f"unexpected answer shape: {obj[:200]!r}")
    if isinstance(data.get("milestones"), list):
        ms = []
        for m in data["milestones"]:
            if not isinstance(m, dict) or not m.get("target"):
                raise RuntimeError(f"bad milestone: {m!r}")
            ms.append({"target": str(m["target"]), "destination": str(m.get("destination") or "")})
        return {"milestones": ms, "say": str(data.get("say", ""))}
    if data.get("done"):
        return {"done": True, "say": str(data.get("say", ""))}
    if data.get("impossible"):
        return {"impossible": str(data["impossible"]), "say": str(data.get("say", ""))}
    nxt = data.get("next")
    if isinstance(nxt, dict) and nxt.get("action") in ("pick_place", "home", "reset"):
        if nxt["action"] == "pick_place" and not nxt.get("target"):
            raise RuntimeError(f"move without a target: {obj[:200]!r}")
        return {"next": nxt, "say": str(data.get("say", "")), "raw": {"say": str(data.get("say", "")), "next": nxt}}
    raise RuntimeError(f"answer has no move, done or impossible: {obj[:200]!r}")


def warm_up() -> Optional[float]:
    """One tiny request at start-up so the system prompt's prefix is already cached when the first real prompt arrives.
    Failure is harmless."""
    try:
        t0 = time.time()
        payload = {"model": cfg.NEMOTRON_MODEL, "temperature": 0, "max_tokens": 1,
                   "chat_template_kwargs": {"enable_thinking": False},
                   "messages": [{"role": "system", "content": STEP_SYSTEM_PROMPT},
                                {"role": "user", "content": 'GOAL: "warm-up"\nSTATE:\n- red cube: on the table, free to pick'}]}
        r = httpx.post(f"{cfg.NEMOTRON_BASE_URL}/chat/completions", json=payload,
                       headers=cfg.auth_headers(cfg.NEMOTRON_API_KEY), timeout=30.0)
        return round(time.time() - t0, 2) if r.status_code == 200 else None
    except Exception:
        return None
