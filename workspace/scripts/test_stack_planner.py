"""Offline tests for workspace/stack_planner.py — no simulator, no arm, no network.

Run from the repo root:   python3 workspace/scripts/test_stack_planner.py [fuzz_layouts]

A tiny world model applies each planned step the way the simulator would (tray slot / onto the real pile top / free spot),
and the checks are made on the resulting positions:
  - a covered cube is never picked, a cube is never put onto its own pile;
  - the plan ends with ALL cubes in ONE column on the requested surface (tray / table / anywhere);
  - the number of moves is minimal (N - biggest reachable column on the surface, or N when a new base is needed);
  - closed loop: planning again after every step (and after random failures that drop a cube somewhere) still converges.
"""
import math
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stack_planner as sp  # noqa: E402

NAMES = {"red_cube": "Red Cube", "green_cube": "Green Cube", "blue_cube": "Blue Cube", "yellow_cube": "Yellow Cube"}


def scene_of(cubes):
    return {"objects": {c.key: {"name": c.name, "position": [c.x, c.y, c.z]} for c in cubes}}


def mk(layout):
    """layout: {key: (x, y, z)} -> list[Cube]"""
    return [sp.Cube(k, NAMES.get(k, k), *v) for k, v in layout.items()]


def apply_step(world, step):
    """Apply one planned step to `world` (dict key -> Cube). Returns an error string or None."""
    c = world[step.key]
    cubes = list(world.values())
    if sp._covered(c, cubes):
        return f"picked a COVERED cube: {c.name}"
    if step.kind == "tray":
        x, y = sp.tray_slot(cubes, c)
        c.x, c.y, c.z = x, y, 0.055
    elif step.kind == "free":
        x, y = sp.free_spot(cubes, c)
        if math.isnan(x):
            return "free spot requested but the table has none"
        c.x, c.y, c.z = x, y, 0.025
    else:
        ref = world[step.ref_key]
        if ref is c:
            return "put a cube on itself"
        col = [o for o in cubes if o is not c and math.hypot(o.x - ref.x, o.y - ref.y) < sp.STACK_XY_M and o.z >= ref.z - 1e-6]
        top = max(col, key=lambda o: o.z)
        if top is not ref:
            return f"destination {ref.name} is not the top of its pile (the {top.name} is)"
        c.x, c.y, c.z = top.x, top.y, top.z + sp.CUBE_M
    return None


def one_pile(world, surface):
    cubes = list(world.values())
    cols = sp.columns(cubes)
    if len(cols) != 1:
        return False
    t = sp.in_tray(cols[0][0].x, cols[0][0].y)
    return surface is None or (surface == "tray" and t) or (surface == "table" and not t)


def expected_moves(cubes, surface):
    cols = sp.columns(cubes)
    n = len(cubes)
    if len(cols) == 1 and one_pile({c.key: c for c in cubes}, surface):
        return 0
    ok = [col for col in cols
          if sp.reachable(col[0].x, col[0].y)
          and (surface is None or (surface == "tray") == sp.in_tray(col[0].x, col[0].y))]
    return n - max((len(c) for c in ok), default=0) if ok else n


def run_open_loop(layout, surface, label=""):
    """Plan once, execute the whole plan with no re-planning."""
    cubes = mk(layout)
    plan = sp.plan_stack(scene_of(cubes), surface)
    world = {c.key: c for c in cubes}
    if plan.already:
        assert one_pile(world, surface), f"{label}: said already stacked but is not"
        return plan
    unreachable = [c.name.lower() for c in cubes if not sp.reachable(c.x, c.y)]
    if plan.blocked or unreachable:                      # exact oracle: blocked  <=>  some cube is out of reach
        assert plan.blocked and unreachable and sorted(plan.blocked) == sorted(unreachable), \
            f"{label}: blocked={plan.blocked} but unreachable={unreachable}"
        assert not plan.steps, f"{label}: a blocked plan must not contain steps"
        return plan
    assert plan.steps, f"{label}: no plan produced ({plan.note})"
    keys = [s.key for s in plan.steps]
    assert len(keys) == len(set(keys)), f"{label}: a cube moves twice: {keys}"
    for s in plan.steps:
        err = apply_step(world, s)
        assert err is None, f"{label}: {err}; plan={plan.goals()}"
    assert one_pile(world, surface), f"{label}: ended not as one pile on {surface}: {[(c.name, round(c.x,2), round(c.y,2), round(c.z,3)) for c in world.values()]}"
    want = expected_moves(mk(layout), surface)
    assert len(plan.steps) == want, f"{label}: {len(plan.steps)} moves, expected {want}; plan={plan.goals()}"
    return plan


def run_closed_loop(layout, surface, fail_p=0.0, rng=None, label="", max_iter=14):
    """Plan, execute ONE step, plan again from the new positions (what the agent does); steps may fail (cube dropped on the table)."""
    rng = rng or random.Random(0)
    world = {c.key: c for c in mk(layout)}
    fails = 0
    for _ in range(max_iter):
        plan = sp.plan_stack(scene_of(list(world.values())), surface)
        if plan.already:
            assert one_pile(world, surface), f"{label}: said already stacked but is not"
            return True
        if plan.blocked:                                 # out of reach: the planner refuses instead of looping
            return True
        assert plan.steps, f"{label}: closed loop stuck with no plan: {plan.note}"
        s = plan.steps[0]
        if rng.random() < fail_p:                       # failed step: the cube was dropped somewhere on the table
            c = world[s.key]
            if sp._covered(c, list(world.values())):
                return False
            for _try in range(30):
                x, y = rng.uniform(0.30, 0.64), rng.uniform(-0.12, 0.40)
                if sp.reachable(x, y) and not sp.in_tray(x, y) and all(math.hypot(o.x - x, o.y - y) > 0.07 for o in world.values() if o is not c):
                    c.x, c.y, c.z = x, y, 0.025
                    break
            fails += 1
            continue
        err = apply_step(world, s)
        assert err is None, f"{label}: {err}"
    return one_pile(world, surface)


# ── named scenarios ─────────────────────────────────────────────────────────────────────────────────────────────────
R, G, B = "red_cube", "green_cube", "blue_cube"
Z0, Z1, Z2 = 0.025, 0.075, 0.125
TX, TY = sp.TRAY_CENTER
SCEN = {
    "all separate on the table": {R: (0.50, 0.00, Z0), G: (0.56, 0.10, Z0), B: (0.45, 0.22, Z0)},
    "already one pile on the table": {R: (0.50, 0.00, Z0), G: (0.50, 0.00, Z1), B: (0.50, 0.00, Z2)},
    "already one pile in the tray": {R: (TX, TY, 0.055), G: (TX, TY, 0.105), B: (TX, TY, 0.155)},
    "pile of 3 on the table, wants the tray": {R: (0.50, 0.00, Z0), G: (0.50, 0.00, Z1), B: (0.50, 0.00, Z2)},
    "pile of 3 in the tray, wants the table": {R: (TX, TY, 0.055), G: (TX, TY, 0.105), B: (TX, TY, 0.155)},
    "pile of 2 on table + 1 in tray": {R: (0.50, 0.00, Z0), G: (0.50, 0.00, Z1), B: (TX, TY, 0.055)},
    "pile of 2 in tray + 1 on table": {R: (TX, TY, 0.055), G: (TX, TY, 0.105), B: (0.56, 0.12, Z0)},
    "two cubes nearby (along Y)": {R: (0.50, 0.00, Z0), G: (0.50, 0.075, Z0), B: (0.55, 0.30, Z0)},
    "two cubes nearby (along X)": {R: (0.50, 0.00, Z0), G: (0.565, 0.00, Z0), B: (0.50, 0.30, Z0)},
    "three in a row along Y": {R: (0.50, 0.00, Z0), G: (0.50, 0.08, Z0), B: (0.50, 0.16, Z0)},
    "one cube covered by another, third apart": {R: (0.50, 0.00, Z0), G: (0.50, 0.00, Z1), B: (0.58, 0.20, Z0)},
    "one in tray, two stacked on table": {R: (TX, TY, 0.055), G: (0.50, 0.10, Z0), B: (0.50, 0.10, Z1)},
    "two in the tray side by side": {R: (0.35, TY, 0.055), G: (0.49, TY, 0.055), B: (0.55, 0.15, Z0)},
    "pile near the robot base (unreachable)": {R: (0.30, 0.00, Z0), G: (0.30, 0.00, Z1), B: (0.58, 0.20, Z0)},
    "only two cubes": {R: (0.50, 0.00, Z0), G: (0.56, 0.18, Z0)},
    "four cubes mixed": {R: (0.50, 0.00, Z0), G: (0.50, 0.00, Z1), B: (TX, TY, 0.055), "yellow_cube": (0.58, 0.25, Z0)},
}


def main():
    n_fuzz = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    print("== named scenarios (open loop) ==")
    failed = 0
    for name, layout in SCEN.items():
        for surface in (None, "tray", "table"):
            label = f"{name} | surface={surface}"
            try:
                p = run_open_loop(layout, surface, label)
                print(f"  ok   {label:<78} {'already' if p.already else ('BLOCKED (out of reach): ' + ', '.join(p.blocked)) if p.blocked else str(len(p.steps)) + ' move(s): ' + ' ; '.join(g['target'] + ' ' + g['destination'] for g in p.goals())}")
            except AssertionError as e:
                failed += 1
                print(f"  FAIL {e}")
    print("\n== shared safe-zone rules (Randomize + free spots) ==")
    rz = random.Random(99)
    bad_rules = 0
    for trial in range(5000):                               # Randomize
        n_c = rz.choice((2, 3, 3, 4))
        sites = sp.random_cube_sites(n_c, rz.uniform)
        ok_ = len(sites) == n_c and all(sp.safe_site(x, y) for x, y in sites) and all(
            math.hypot(a[0] - b[0], a[1] - b[1]) >= sp.CUBE_SEP_M - 1e-9 for i_, a in enumerate(sites) for b in sites[i_ + 1:])
        if ok_:                                             # and explicitly: not near the base, not near the tray, in reach
            ok_ = all(sp.SAFE_R[0] <= math.hypot(x, y) <= sp.SAFE_R[1] and sp.tray_distance(x, y) >= sp.TRAY_SAFE_M
                      and sp.reachable(x, y) for x, y in sites)
        if not ok_:
            bad_rules += 1
            if bad_rules <= 3:
                print("  FAIL randomize:", sites)
    print(f"  Randomize: 5000 draws, {bad_rules} violations of safe zone / tray distance / base distance / separation")
    nan_spots = bad_spots = 0
    for trial in range(5000):                               # free spots, with 1-3 other cubes anywhere (also crowded, near tray)
        others = []
        for _ in range(rz.choice((1, 2, 3))):
            others.append((rz.uniform(0.25, 0.66), rz.uniform(-0.40, 0.42)))
        src = (rz.uniform(0.28, 0.65), rz.uniform(-0.15, 0.42))
        spot = sp.find_free_spot(others, src)
        if spot is None:
            nan_spots += 1                                  # allowed only when the table is genuinely too crowded
            continue
        x, y = spot
        room = min(math.hypot(x - ox, y - oy) for ox, oy in others)
        if not (sp.safe_site(x, y) and room >= sp.FREE_SPOT_MIN_CLEARANCE_M - 1e-9):
            bad_spots += 1
            if bad_spots <= 3:
                print("  FAIL free spot", spot, "others", others)
    print(f"  Free spots: 5000 searches, {bad_spots} unsafe results, {nan_spots} refusals (table too crowded)")
    failed += bad_rules + bad_spots
    print("\n== fuzz ==")
    rng = random.Random(12345)

    def rand_xy(avoid, tray=False):
        for _ in range(400):
            if tray:
                x, y = rng.uniform(TX - 0.06, TX + 0.06), rng.uniform(TY - 0.07, TY + 0.07)
            else:
                x, y = rng.uniform(0.28, 0.65), rng.uniform(-0.15, 0.42)
                if rng.random() < 0.08:
                    x, y = rng.uniform(0.25, 0.36), rng.uniform(-0.1, 0.2)         # sometimes unreachably close
                if sp.in_tray(x, y) or math.hypot(x, y) > 0.66:
                    continue
            if all(math.hypot(x - ax, y - ay) > 0.062 for ax, ay in avoid):
                return x, y
        return None

    def gen():
        n = rng.choice((2, 3, 3, 3, 4))
        keys = [R, G, B, "yellow_cube"][:n]
        rng.shuffle(keys)
        sizes, left = [], n
        while left:
            s = rng.randint(1, left)
            sizes.append(s)
            left -= s
        layout, used, i = {}, [], 0
        for s in sizes:
            xy = rand_xy(used, tray=(rng.random() < 0.35))
            if xy is None:
                return None
            used.append(xy)
            for lvl in range(s):
                x, y = xy[0] + rng.gauss(0, 0.003), xy[1] + rng.gauss(0, 0.003)
                base_z = 0.055 if sp.in_tray(*xy) else 0.025
                layout[keys[i]] = (x, y, base_z + 0.05 * lvl + rng.gauss(0, 0.002))
                i += 1
        return layout

    count = bad = loops_bad = 0
    stats = {"moves": 0, "already": 0}
    while count < n_fuzz:
        layout = gen()
        if layout is None:
            continue
        count += 1
        for surface in (None, "tray", "table"):
            label = f"fuzz#{count} surface={surface} layout={ {k: tuple(round(v, 3) for v in p) for k, p in layout.items()} }"
            try:
                p = run_open_loop(layout, surface, label)
                stats["already" if p.already else "moves"] += 1
                assert run_closed_loop(layout, surface, 0.0, rng, label), f"{label}: closed loop did not converge"
                if not run_closed_loop(layout, surface, 0.25, rng, label, max_iter=24):
                    loops_bad += 1
            except AssertionError as e:
                bad += 1
                if bad <= 8:
                    print("  FAIL", e)
    print(f"\n{count} random layouts x 3 surfaces: {bad} failures, {stats['already']} already-stacked, {stats['moves']} planned; "
          f"closed loop with 25% random step failures did not converge in {loops_bad} runs")
    total_bad = failed + bad
    print("RESULT:", "ALL PASSED" if total_bad == 0 and loops_bad == 0 else f"{total_bad} FAILED, {loops_bad} non-converging")
    return 1 if total_bad or loops_bad else 0


if __name__ == "__main__":
    sys.exit(main())
