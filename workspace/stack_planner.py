"""Local, deterministic stacking planner (no LLM, no simulator imports).

Input: the live scene (cube positions, as the agent sees them) and the surface the pile must stand on.
Output: an ordered list of pick-and-place subgoals that ends with ALL cubes in ONE vertical pile on that surface.

How it decides (everything is computed from positions, never from colour):

  1. Columns.  Cubes whose xy are within STACK_XY_M and which sit one above the other form a column (bottom -> top).
     A cube is COVERED when something is on top of it; a covered cube can never be picked, so columns are always
     taken apart from the TOP.
  2. Pile site.  Candidates are (a) an existing column on the requested surface that is reachable (the base stays put) or,
     when there is none, (b) a NEW base: an uncovered cube carried into the tray / onto a free table spot.  The plan
     with the fewest moves wins; ties go to the cheapest one (total carry distance, hand-collision risk, pile not too
     close to the robot).
  3. Order.  Every cube outside the pile is carried onto the pile top.  A cube is only picked once it is uncovered;
     among the valid orders the cheapest is used, with the nearest cube first on ties.  The cost includes a hand
     collision term: the open fingers straddle a cube along world Y, so a neighbour sitting beside it along Y (or beside
     the pile along Y) makes that pick/place risky; orders that avoid such picks are preferred.
  4. Surfaces.  "tray": the base stands in the tray.  "table": the pile stands on the table, never in the tray.
     None: wherever the biggest reachable column already is.

`plan_stack` is called again after every executed step (closed loop), so a step that failed or knocked something over
simply produces a new plan from the real state.
"""
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

CUBE_M = 0.05
STACK_XY_M = 0.035                 # |dxy| below this = the other cube sits on this one
STACK_MIN_DZ_M = 0.035             # ...and is at least this much higher (a full layer is CUBE_M)
TRAY_CENTER = (0.42, -0.32)
TRAY_HALF = (0.11, 0.12)
TRAY_MARGIN_M = 0.025              # a cube whose centre is this close to the slab's edge still counts as in the tray
TRAY_SLOTS = ((0.42, -0.32), (0.35, -0.32), (0.49, -0.32))
TRAY_SLOT_OCCUPIED_M = 0.055
REACH_MIN_M, REACH_MAX_M = 0.38, 0.66      # radius from the robot base at which the downward hand can grasp
WORKSPACE_X = (0.20, 0.70)        # the agent's guardrail box
WORKSPACE_Y = (-0.45, 0.45)
NEAR_BASE_M = 0.40                 # a pile closer than this needs the wrist near its limit: mildly penalised
FINGER_REACH_Y_M = 0.09            # open fingers reach this far from the cube centre along world Y (yaw 0)
FINGER_HALF_X_M = 0.045            # ...and a neighbour this close across that axis is hit
# Safe zone: everywhere the SYSTEM places a cube (Randomize, "a free spot on the table"). Stricter than the hard reach limits
# above so a cube is never put where the hand is near its joint limits, close to the robot base, hugging the tray or at the
# edge of the table.
SAFE_R = (0.42, 0.62)              # radius from the robot base: not close to the base, comfortably inside reach
SAFE_X = (0.30, 0.64)
SAFE_Y = (-0.15, 0.40)
TRAY_SAFE_M = 0.12                 # a cube centre is at least this far from the tray slab (the open hand is ~0.11 m wide)
CUBE_SEP_M = 0.12                  # Randomize keeps cube centres this far apart
FREE_SPOT_CLEARANCE_M = 0.15       # a free spot is this far from every other cube (relaxed in steps to FREE_SPOT_MIN_CLEARANCE_M)
FREE_SPOT_MIN_CLEARANCE_M = 0.11
COMFORT_R = 0.52                   # the radius at which the downward hand is most comfortable
HAZARD_W = 3.0
MAX_EXACT = 6                      # exhaustive ordering up to this many cubes to move; greedy beyond

FREE_SPOT_DESTINATION = "on a free spot on the table"


@dataclass
class Cube:
    key: str
    name: str
    x: float
    y: float
    z: float


@dataclass
class Step:
    key: str                       # cube to move
    name: str
    kind: str                      # "tray" | "on" | "free"
    ref_key: Optional[str] = None  # for "on": the cube it goes onto (the current top of the pile)
    ref_name: Optional[str] = None

    def destination(self) -> str:
        if self.kind == "tray":
            return "in the tray"
        if self.kind == "free":
            return FREE_SPOT_DESTINATION
        return f"on top of the {self.ref_name}"


@dataclass
class StackPlan:
    steps: List[Step] = field(default_factory=list)
    already: bool = False          # the cubes already form one pile on the requested surface
    note: str = ""
    blocked: List[str] = field(default_factory=list)   # cubes that cannot be picked or stacked on (out of reach)
    mode: str = ""                 # "existing" | "new_tray" | "new_free" | ""

    def goals(self) -> List[Dict[str, Any]]:
        return [{"action": "pick_place", "target": s.name, "destination": s.destination()} for s in self.steps]


# ── geometry ────────────────────────────────────────────────────────────────────────────────────────────────────────

def cubes_from_scene(scene: Dict[str, Any]) -> List[Cube]:
    out = []
    for key, o in (scene.get("objects") or {}).items():
        if not key.endswith("_cube") or not o.get("position"):
            continue
        p = o["position"]
        out.append(Cube(key, str(o.get("name", key)).lower(), float(p[0]), float(p[1]), float(p[2])))
    return out


def in_tray(x: float, y: float) -> bool:
    return (abs(x - TRAY_CENTER[0]) <= TRAY_HALF[0] + TRAY_MARGIN_M and
            abs(y - TRAY_CENTER[1]) <= TRAY_HALF[1] + TRAY_MARGIN_M)


def reachable(x: float, y: float) -> bool:
    return (REACH_MIN_M <= math.hypot(x, y) <= REACH_MAX_M and WORKSPACE_X[0] <= x <= WORKSPACE_X[1]
            and WORKSPACE_Y[0] <= y <= WORKSPACE_Y[1])


def _xy(c: Cube) -> Tuple[float, float]:
    return (c.x, c.y)


def columns(cubes: List[Cube]) -> List[List[Cube]]:
    """Cubes grouped into stacks, each ordered bottom -> top."""
    cols: List[List[Cube]] = []
    for c in sorted(cubes, key=lambda c: c.z):
        for col in cols:
            if math.hypot(c.x - col[0].x, c.y - col[0].y) < STACK_XY_M and c.z - col[-1].z > STACK_MIN_DZ_M:
                col.append(c)
                break
        else:
            cols.append([c])
    return cols


def _covered(c: Cube, cubes: List[Cube]) -> bool:
    return any(o is not c and math.hypot(o.x - c.x, o.y - c.y) < STACK_XY_M and o.z - c.z > STACK_MIN_DZ_M for o in cubes)


def _hazard(px: float, py: float, cubes: List[Cube], ignore_xy_of: Optional[Tuple[float, float]] = None,
            exclude: Optional[Cube] = None) -> int:
    """Cubes the open fingers would hit when working at (px, py): neighbours beside it along Y. Cubes of the same column
    (within STACK_XY_M of the spot) are not neighbours."""
    n = 0
    for o in cubes:
        if o is exclude:
            continue
        if math.hypot(o.x - px, o.y - py) < STACK_XY_M:
            continue
        if abs(o.y - py) < FINGER_REACH_Y_M and abs(o.x - px) < FINGER_HALF_X_M:
            n += 1
    return n


def tray_distance(x: float, y: float) -> float:
    """Distance (m) from a table point to the tray slab's footprint (0 on it)."""
    dx = max(TRAY_CENTER[0] - TRAY_HALF[0] - x, 0.0, x - TRAY_CENTER[0] - TRAY_HALF[0])
    dy = max(TRAY_CENTER[1] - TRAY_HALF[1] - y, 0.0, y - TRAY_CENTER[1] - TRAY_HALF[1])
    return math.hypot(dx, dy)


def safe_site(x: float, y: float) -> bool:
    """Inside the safe zone: far enough from the robot base, well inside reach, clear of the tray and the table edge."""
    return (SAFE_R[0] <= math.hypot(x, y) <= SAFE_R[1] and SAFE_X[0] <= x <= SAFE_X[1] and SAFE_Y[0] <= y <= SAFE_Y[1]
            and tray_distance(x, y) >= TRAY_SAFE_M)


def find_free_spot(others_xy, from_xy=None, step: float = 0.02):
    """The best free table spot for a cube: inside the safe zone, at least FREE_SPOT_CLEARANCE_M from every other cube
    (relaxed in steps down to FREE_SPOT_MIN_CLEARANCE_M if the table is crowded), and cheap: the shortest carry from
    `from_xy`, preferring more room around it and a comfortable radius.  Returns (x, y) or None when there is no such spot."""
    others_xy = list(others_xy)
    grid = []
    x = SAFE_X[0]
    while x <= SAFE_X[1] + 1e-9:
        y = SAFE_Y[0]
        while y <= SAFE_Y[1] + 1e-9:
            if safe_site(x, y):
                grid.append((x, y))
            y += step
        x += step
    for clearance in (FREE_SPOT_CLEARANCE_M, 0.13, FREE_SPOT_MIN_CLEARANCE_M):
        best = None
        for gx, gy in grid:
            room = min([math.hypot(gx - ox, gy - oy) for ox, oy in others_xy] or [1.0])
            if room < clearance:
                continue
            d = math.hypot(gx - from_xy[0], gy - from_xy[1]) if from_xy else 0.0
            cost = d + 0.5 * max(0.0, 0.22 - room) + 0.25 * abs(math.hypot(gx, gy) - COMFORT_R)
            if best is None or cost < best[0] - 1e-12:
                best = (cost, (round(gx, 4), round(gy, 4)))
        if best:
            return best[1]
    return None


def random_cube_sites(n: int, uniform, sep: float = CUBE_SEP_M, tries: int = 400):
    """`n` random cube positions for Randomize: every one inside the safe zone and at least `sep` from the others.
    `uniform(a, b)` is the caller's random source (np.random.uniform / random.Random.uniform)."""
    sites = []
    for _ in range(n):
        for _t in range(tries):
            x, y = float(uniform(*SAFE_X)), float(uniform(*SAFE_Y))
            if safe_site(x, y) and all(math.hypot(x - sx, y - sy) >= sep for sx, sy in sites):
                sites.append((x, y))
                break
        else:                                            # rejection sampling failed: take the best remaining spot
            spot = find_free_spot(sites, None)
            if spot is None:
                break
            sites.append(spot)
    return sites


def free_spot(cubes: List[Cube], moving: Cube) -> Tuple[float, float]:
    """The spot the simulator picks for 'a free spot on the table' (it uses find_free_spot too)."""
    spot = find_free_spot([(o.x, o.y) for o in cubes if o is not moving], (moving.x, moving.y))
    return spot if spot else (float("nan"), float("nan"))


def tray_slot(cubes: List[Cube], moving: Cube) -> Tuple[float, float]:
    for sx, sy in TRAY_SLOTS:
        if all(math.hypot(o.x - sx, o.y - sy) >= TRAY_SLOT_OCCUPIED_M for o in cubes if o is not moving):
            return (sx, sy)
    return TRAY_SLOTS[0]


# ── planning ────────────────────────────────────────────────────────────────────────────────────────────────────────

def _order_cost(movers: List[Cube], pile_xy: Tuple[float, float], pile_top: Cube, all_cubes: List[Cube],
                new_base: Optional[Cube] = None) -> Optional[Tuple[float, List[Cube]]]:
    """Cheapest valid order to carry `movers` onto the pile, or None if no order lets every cube be picked uncovered."""
    best = None
    perms = itertools.permutations(movers) if len(movers) <= MAX_EXACT else [tuple(_greedy_order(movers, pile_xy))]
    for perm in perms:
        cur = [Cube(c.key, c.name, c.x, c.y, c.z) for c in all_cubes]
        by_key = {c.key: c for c in cur}
        top_z = pile_top.z
        cost, ok = 0.0, True
        for t, mv in enumerate(perm):
            c = by_key[mv.key]
            if _covered(c, cur):
                ok = False
                break
            dist = math.hypot(c.x - pile_xy[0], c.y - pile_xy[1])
            cost += dist + 1e-3 * (len(perm) - t) * dist
            cost += HAZARD_W * _hazard(c.x, c.y, cur, exclude=c)
            cost += HAZARD_W * _hazard(pile_xy[0], pile_xy[1], cur, exclude=c)
            top_z += CUBE_M
            c.x, c.y, c.z = pile_xy[0], pile_xy[1], top_z
        if ok and (best is None or cost < best[0] - 1e-9):
            best = (cost, [by_key_orig for by_key_orig in perm])
    return best


def _greedy_order(movers: List[Cube], pile_xy) -> List[Cube]:
    return sorted(movers, key=lambda c: math.hypot(c.x - pile_xy[0], c.y - pile_xy[1]))


def _pile_penalty(x: float, y: float) -> float:
    r = math.hypot(x, y)
    return (1.0 if r < NEAR_BASE_M else 0.0) + 0.05 * r


def plan_stack(scene: Dict[str, Any], surface: Optional[str] = None) -> StackPlan:
    """The plan that turns the current scene into one pile on `surface` ('tray', 'table' or None = wherever)."""
    cubes = cubes_from_scene(scene)
    if len(cubes) < 2:
        return StackPlan(note="(fewer than two cubes are visible - nothing to stack)")
    cols = columns(cubes)
    n = len(cubes)

    # already one pile on the requested surface?
    if len(cols) == 1:
        base = cols[0][0]
        base_in_tray = in_tray(base.x, base.y)
        if surface is None or (surface == "tray" and base_in_tray) or (surface == "table" and not base_in_tray):
            return StackPlan(already=True, note="(the cubes are already stacked in one pile"
                                                 + (" in the tray" if surface == "tray" else " on the table" if surface == "table" else "")
                                                 + " - nothing to do)")

    def on_surface(col):
        t = in_tray(col[0].x, col[0].y)
        return surface is None or (surface == "tray" and t) or (surface == "table" and not t)

    # A cube outside the reachable zone can neither be picked nor stacked on (the guardrail refuses both), and every
    # cube has to end up in the pile, so the pile would have to stand at an unreachable spot: nothing can be done.
    out_of_reach = [c.name for c in cubes if not reachable(c.x, c.y)]
    if out_of_reach:
        return StackPlan(blocked=out_of_reach, note="(cannot stack: the " + " and the ".join(out_of_reach)
                         + " cannot be reached - too close to the robot or outside its working area - so it cannot be picked up "
                         "or stacked on. Move it with 'randomize' or 'reset' first.)")

    candidates = []                                    # (n_moves, cost, mode, payload)
    # (a) keep an existing column as the pile (its base stays where it is)
    for col in cols:
        if not on_surface(col) or not reachable(col[0].x, col[0].y):
            continue
        movers = [c for c in cubes if c not in col]
        pile_xy = (col[0].x, col[0].y)
        res = _order_cost(movers, pile_xy, col[-1], cubes) if movers else (0.0, [])
        if res is None:
            continue
        cost = res[0] + _pile_penalty(*pile_xy)
        candidates.append((len(movers), cost, "existing", (col, res[1])))
    # (b) a new base carried into the tray / onto a free spot (only when no usable column exists on that surface)
    if not candidates:
        mode = "new_tray" if surface == "tray" else "new_free"
        for b in cubes:
            if _covered(b, cubes):
                continue
            if mode == "new_tray":
                pile_xy = tray_slot(cubes, b)
            else:
                pile_xy = free_spot(cubes, b)
                if math.isnan(pile_xy[0]):
                    continue
            others = [c for c in cubes if c is not b]
            # after the base has moved it sits at pile_xy, one layer high; model it as the pile top for the others
            moved_base = Cube(b.key, b.name, pile_xy[0], pile_xy[1], CUBE_M / 2)
            world = [c if c is not b else moved_base for c in cubes]
            res = _order_cost([w for w in world if w.key != b.key], pile_xy, moved_base, world) if others else (0.0, [])
            if res is None:
                continue
            cost = math.hypot(b.x - pile_xy[0], b.y - pile_xy[1]) + HAZARD_W * _hazard(b.x, b.y, cubes, exclude=b) \
                + HAZARD_W * _hazard(pile_xy[0], pile_xy[1], world, exclude=moved_base) + res[0] + _pile_penalty(*pile_xy)
            candidates.append((n, cost, mode, (b, pile_xy, res[1])))
    if not candidates:
        return StackPlan(note="(no way to build a pile from the current positions was found)")

    candidates.sort(key=lambda c: (c[0], c[1]))
    n_moves, cost, mode, payload = candidates[0]
    steps: List[Step] = []
    notes: List[str] = []
    cur = {c.key: c for c in cubes}
    if mode == "existing":
        col, order = payload
        top = col[-1]
        notes.append(f"pile stays where the {col[0].name} already stands" + (f" ({len(col)} cubes high)" if len(col) > 1 else ""))
    else:
        b, pile_xy, order = payload
        kind = "tray" if mode == "new_tray" else "free"
        steps.append(Step(b.key, b.name, kind))
        top = b
        notes.append(f"the {b.name} is carried {'into the tray' if kind == 'tray' else 'to a free spot on the table'} as the base"
                     + (" (the pile there is taken apart from the top)" if len(cols) < n else ""))
    for mv in order:
        steps.append(Step(mv.key, mv.name, "on", top.key, top.name))
        top = mv
    order_txt = " -> ".join(s.name for s in steps)
    covered = [c.name for c in cubes if _covered(c, cubes)]
    if covered:
        notes.append("covered cubes (" + ", ".join(covered) + ") are never picked until the cube on top has been moved")
    haz = sum(1 for c in cubes if _hazard(c.x, c.y, cubes, exclude=c))
    if haz:
        notes.append("cubes that sit beside another along the finger axis are ordered to keep the hand clear where possible")
    note = (f"(local stack planner, from the live positions, not by colour: {'; '.join(notes)}. "
            f"Order: {order_txt}, {len(steps)} move(s).)")
    return StackPlan(steps=steps, note=note, mode=mode)


def is_stacked(scene: Dict[str, Any], surface: Optional[str] = None) -> bool:
    return plan_stack(scene, surface).already
