"""Visual-only factory backdrop for the Isaac Sim scene (aesthetics only).

Nothing here has physics or collision (VisualCuboid / VisualCylinder), and nothing touches the arm, the work table, the
tray or the cubes, so picking, planning and perception behave exactly as before. Everything is built under /World/Factory so
it can be removed and rebuilt in one go (POST /background/rebuild hot-reloads this module without restarting the sim).

Layout is chosen for the overview camera at (1.85, -1.55, 1.30) looking at (0.35, 0, 0.28): its background is the back-right
corner of the room, so the back wall (x = BACK_X) and the right wall (y = RIGHT_Y) carry the scenery. The robot stands at the
origin; the work table top is Z = 0, spans x 0..0.9, y -0.65..0.65.

Colours: mostly neutral greys. The cube detector looks for saturated red / green / blue, but only inside the table region of
the image, and the few saturated accents below (blue bins, signal tower) are well outside it.
"""
import math

import numpy as np

ROOT = "/World/Factory"
FLOOR_Z = -0.85          # visual floor; the (invisible) physics ground plane is left exactly where it was
BACK_X = -3.40           # back wall, behind the robot
RIGHT_Y = 3.60           # right-hand wall
LEFT_Y = -2.60
CEIL_Z = 3.70
CEILING = True           # a ceiling with beams and light fixtures: darker, moodier room (the calibration adapts, see color_segment)
SCENERY_SCALE = 0.5     # the overview camera only sees the bottom ~1.4 m of the far walls, so wall scenery is built half size

WHITE = (0.92, 0.93, 0.94)
OFFWHITE = (0.90, 0.91, 0.92)
LIGHTGREY = (0.72, 0.74, 0.76)
MIDGREY = (0.48, 0.50, 0.53)
DARKGREY = (0.17, 0.18, 0.20)
BLACK = (0.05, 0.05, 0.06)
STEEL = (0.66, 0.68, 0.71)
CONCRETE = (0.50, 0.51, 0.53)
YELLOW = (0.93, 0.74, 0.05)
CARDBOARD = (0.60, 0.46, 0.30)
BLUEBIN = (0.10, 0.22, 0.48)
BINGREY = (0.40, 0.42, 0.44)


def build(stage):
    """(Re)builds the whole backdrop. Safe to call again: the previous copy is removed first."""
    from isaacsim.core.api.objects import VisualCuboid, VisualCylinder
    from isaacsim.core.utils.rotations import euler_angles_to_quat

    if stage.GetPrimAtPath(ROOT).IsValid():
        stage.RemovePrim(ROOT)
    counter = [0]
    sc = [1.0]                                              # current scenery scale about the room's back-right floor corner

    def W(x, y):
        """Where a scenery-space (x, y) lands in the world once the scenery scale is applied."""
        return (BACK_X + SCENERY_SCALE * (x - BACK_X), RIGHT_Y + SCENERY_SCALE * (y - RIGHT_Y))
    pivot = np.array([BACK_X, RIGHT_Y, FLOOR_Z], dtype=float)

    def _p(c):
        return pivot + sc[0] * (np.array(c, dtype=float) - pivot) if sc[0] != 1.0 else np.array(c, dtype=float)

    def _name(prefix):
        counter[0] += 1
        return f"{prefix}_{counter[0]}"

    def box(center, size, color, rot=None):
        """Visual cuboid; `size` = full extents (m); `rot` = (rx, ry, rz) degrees."""
        n = _name("b")
        kw = {}
        if rot is not None:
            kw["orientation"] = euler_angles_to_quat(np.radians(np.array(rot, dtype=float)))
        VisualCuboid(prim_path=f"{ROOT}/{n}", name=f"fac_{n}", position=_p(center),
                     scale=np.array(size, dtype=float) * sc[0], color=np.array(color, dtype=float), **kw)

    def cyl(center, radius, height, color, rot=None):
        n = _name("c")
        kw = {}
        if rot is not None:
            kw["orientation"] = euler_angles_to_quat(np.radians(np.array(rot, dtype=float)))
        VisualCylinder(prim_path=f"{ROOT}/{n}", name=f"fac_{n}", position=_p(center),
                       radius=float(radius) * sc[0], height=float(height) * sc[0], color=np.array(color, dtype=float), **kw)

    fz = FLOOR_Z
    wall_h = CEIL_Z - fz
    wall_zc = (CEIL_Z + fz) / 2.0
    room_x0, room_x1 = BACK_X, 3.6
    room_cx, room_lx = (room_x0 + room_x1) / 2.0, room_x1 - room_x0
    room_cy, room_ly = (LEFT_Y + RIGHT_Y) / 2.0, RIGHT_Y - LEFT_Y

    # ── room shell ─────────────────────────────────────────────────────────────────────────────────────────────────
    box((room_cx, room_cy, fz - 0.01), (room_lx, room_ly, 0.02), CONCRETE)                      # polished concrete floor
    box((BACK_X - 0.03, room_cy, wall_zc), (0.06, room_ly, wall_h), OFFWHITE)                   # back wall
    box((room_cx, RIGHT_Y + 0.03, wall_zc), (room_lx, 0.06, wall_h), OFFWHITE)                  # right wall
    box((room_cx, LEFT_Y - 0.03, wall_zc), (room_lx, 0.06, wall_h), OFFWHITE)                   # left wall
    box((BACK_X + 0.02, room_cy, fz + 0.60), (0.04, room_ly, 1.20), (0.62, 0.64, 0.67))         # darker lower wall band, back
    box((room_cx, RIGHT_Y - 0.02, fz + 0.60), (room_lx, 0.04, 1.20), (0.62, 0.64, 0.67))        # ... and right
    box((BACK_X + 0.03, room_cy, fz + 0.06), (0.06, room_ly, 0.12), DARKGREY)                   # skirting
    box((room_cx, RIGHT_Y - 0.03, fz + 0.06), (room_lx, 0.06, 0.12), DARKGREY)

    # ── ceiling, beams and strip-light fixtures (fixtures only; the scene's calibrated lights are untouched) ────────
    if CEILING:
        box((room_cx, room_cy, CEIL_Z + 0.03), (room_lx, room_ly, 0.06), (0.60, 0.62, 0.65))
        for cx_ in (-2.0, -0.8, 0.4):
            box((cx_, room_cy, CEIL_Z - 0.14), (0.20, room_ly, 0.20), MIDGREY)
        for lx in (-2.6, -1.4, -0.2, 1.0):
            for ly in (0.0, 1.4, 2.8):
                box((lx, ly, CEIL_Z - 0.04), (0.16, 1.10, 0.05), (1.0, 1.0, 1.0))

    # ── hazard-stripe floor tape (yellow / black dashes) ────────────────────────────────────────────────────────────
    def hazard_line(p0, p1, width=0.12, dash=0.16):
        p0, p1 = np.array(p0, float), np.array(p1, float)
        length = float(np.linalg.norm(p1 - p0))
        ang = math.degrees(math.atan2(p1[1] - p0[1], p1[0] - p0[0]))
        n = max(1, int(length / dash))
        for i in range(n):
            c = p0 + (p1 - p0) * ((i + 0.5) / n)
            box((c[0], c[1], fz + 0.003), (length / n, width, 0.006), YELLOW if i % 2 == 0 else BLACK, rot=(0, 0, ang))
    hazard_line((1.50, -2.4, 0), (1.50, 0.9, 0))
    hazard_line((1.50, 0.80, 0), (-3.10, 0.80, 0))

    # ── work-table frame: visual rear extension so the robot base sits on the table, plus steel legs and rails ───
    # The real table (FixedCuboid, x 0..0.9) is not touched; this slab is visual only, same height and tone, so the arm
    # base (at x = 0) no longer overhangs the edge.
    TX0, TX1 = -0.35, 0.90                                                                      # table extent incl. the extension
    box(((TX0 + 0.0) / 2.0, 0.0, -0.05), (0.0 - TX0, 1.30, 0.10), (0.16, 0.18, 0.22))
    legx = (TX0 + 0.05, TX1 - 0.05)
    for lx in legx:
        for ly in (-0.62, 0.62):
            box((lx, ly, (fz - 0.10) / 2.0), (0.07, 0.07, -0.10 - fz), BLACK)
            box((lx, ly, fz + 0.02), (0.14, 0.14, 0.04), DARKGREY)
    for lx in legx:
        box((lx, 0.0, fz + 0.30), (0.04, 1.24, 0.05), DARKGREY)
    for ly in (-0.62, 0.62):
        box(((legx[0] + legx[1]) / 2.0, ly, fz + 0.30), (legx[1] - legx[0], 0.04, 0.05), DARKGREY)

    sc[0] = SCENERY_SCALE
    # ── back wall: whiteboard, pegboard, tool cabinet ───────────────────────────────────────────────────────────────
    wx = BACK_X + 0.03
    WY = -1.15                                                                                  # whiteboard centre (along y)
    box((wx, WY, 1.20), (0.04, 1.90, 1.05), DARKGREY)                                           # whiteboard frame
    box((wx + 0.02, WY, 1.20), (0.02, 1.82, 0.97), (0.97, 0.97, 0.98))                          # whiteboard
    box((wx + 0.02, WY, 0.68), (0.05, 1.82, 0.03), MIDGREY)                                     # marker tray
    ink = (0.10, 0.10, 0.12)
    for i, ww in enumerate((0.34, 0.44, 0.30, 0.40)):                                           # bullet list scribbles
        zz = 1.52 - i * 0.19
        box((wx + 0.034, WY + 0.32 + ww / 2, zz), (0.006, ww, 0.03), ink)
        box((wx + 0.034, WY + 0.22, zz), (0.006, 0.045, 0.045), ink)
    for k, yy in enumerate((WY - 0.40, WY - 0.80)):                                             # two cube sketches
        box((wx + 0.034, yy, 1.30), (0.006, 0.24, 0.016), ink)
        box((wx + 0.034, yy, 1.12), (0.006, 0.24, 0.016), ink)
        box((wx + 0.034, yy - 0.12, 1.21), (0.006, 0.016, 0.18), ink)
        box((wx + 0.034, yy + 0.12, 1.21), (0.006, 0.016, 0.18), ink)
    PY = WY + 1.75                                                                              # pegboard to the right of the board
    box((wx + 0.02, PY, 0.95), (0.04, 1.30, 0.90), (0.12, 0.13, 0.15))
    for i in range(7):
        box((wx + 0.07, PY - 0.55 + i * 0.18, 0.88), (0.07, 0.04, 0.34), (0.55, 0.12, 0.10) if i % 3 == 0 else STEEL)
    for i in range(4):                                                                          # yellow parts bins on the pegboard
        box((wx + 0.10, PY - 0.45 + i * 0.30, 1.20), (0.12, 0.22, 0.10), YELLOW)
    CY = WY                                                                                     # tool cabinet under the whiteboard
    box((BACK_X + 0.34, CY, fz + 0.50), (0.60, 1.40, 1.00), (0.10, 0.11, 0.13))
    box((BACK_X + 0.34, CY, fz + 1.015), (0.62, 1.44, 0.03), (0.20, 0.21, 0.23))
    for i in range(5):
        zz = fz + 0.10 + i * 0.19
        box((BACK_X + 0.645, CY, zz), (0.012, 1.32, 0.17), (0.16, 0.17, 0.19))
        box((BACK_X + 0.66, CY, zz + 0.045), (0.02, 0.60, 0.02), STEEL)
    for yy in (CY - 0.60, CY + 0.60):                                                           # castors
        for xx in (BACK_X + 0.12, BACK_X + 0.56):
            cyl((xx, yy, fz + 0.04), 0.04, 0.08, BLACK, rot=(90, 0, 0))

    # ── industrial racking: steel frames, shelves, cardboard boxes, blue bins ───────────────────────────────────────
    def rack(x, y, width, depth, levels, height, seed):
        rs = np.random.RandomState(seed)
        for dx in (-width / 2, width / 2):
            for dy in (-depth / 2, depth / 2):
                box((x + dy, y + dx, fz + height / 2), (0.05, 0.05, height), (0.12, 0.13, 0.15))
        for lv in range(levels):
            zz = fz + 0.20 + lv * (height - 0.30) / max(1, levels - 1)
            box((x, y, zz), (depth, width, 0.04), (0.78, 0.45, 0.10) if lv % 2 else (0.20, 0.21, 0.24))
            cx = -width / 2 + 0.16
            while cx < width / 2 - 0.16:
                bw = float(rs.uniform(0.20, 0.34))
                if cx + bw > width / 2 - 0.03:
                    break
                bh = float(rs.uniform(0.18, 0.30))
                kind = rs.rand()
                col = CARDBOARD if kind < 0.55 else (BLUEBIN if kind < 0.8 else (0.70, 0.71, 0.73))
                box((x, y + cx + bw / 2, zz + 0.02 + bh / 2), (depth * 0.82, bw, bh), col)
                cx += bw + 0.05
    rack(BACK_X + 0.32, 1.85, 1.50, 0.60, 4, 2.40, 1)

    # ── conveyor along the right wall: one continuous run that leaves the frame on the right ───────────────────────
    cy0, ctop = RIGHT_Y - 0.55, fz + 0.88                    # centre line (y) and belt height
    cx_a, cx_end = -3.10, 6.00                                # starts at the back-wall corner, runs far past the right edge
    STEELC, ZINC, BELT = (0.58, 0.60, 0.63), (0.74, 0.76, 0.78), (0.09, 0.09, 0.10)
    DARKL = (0.16, 0.17, 0.19)

    def conveyor_leg(axis, line, a0, a1, rollers=(True, True), drive=False, items=()):
        """One straight section from a0 to a1 along world X (axis 'x', centre line y = line) or world Y (axis 'y', x = line):
        side frame with guard rail, rubber belt with splice lines, end rollers, legs with braces, optional drive motor,
        and cartons / totes riding the belt."""
        L, mid = a1 - a0, (a0 + a1) / 2.0

        def P(a, c, z):                                       # position along the belt (a), across it (c), height z
            return (a, line + c, z) if axis == "x" else (line + c, a, z)

        def Z(length, width, h):                              # box size: length along the belt, width across it
            return (length, width, h) if axis == "x" else (width, length, h)
        across = (90, 0, 0) if axis == "x" else (0, 90, 0)    # cylinder axis across the belt
        along = (0, 90, 0) if axis == "x" else (90, 0, 0)     # cylinder axis along the belt
        for sy in (-1, 1):                                    # side frame: channel + top guard lip
            box(P(mid, sy * 0.31, ctop - 0.07), Z(L, 0.04, 0.16), STEELC)
            box(P(mid, sy * 0.285, ctop + 0.005), Z(L, 0.015, 0.03), ZINC)
        trim = sum(1 for r in rollers if r) * 0.05
        box(P(mid + (0.05 if rollers[0] and not rollers[1] else -0.05 if rollers[1] and not rollers[0] else 0), 0, ctop - 0.015),
            Z(L - trim, 0.56, 0.02), BELT)                                                       # rubber belt
        for k in range(int(L / 0.55)):                                                           # belt splice lines
            box(P(a0 + 0.30 + k * 0.55, 0, ctop - 0.003), Z(0.012, 0.54, 0.004), (0.17, 0.17, 0.19))
        for a, has in ((a0 + 0.02, rollers[0]), (a1 - 0.02, rollers[1])):                        # end rollers
            if has:
                cyl(P(a, 0, ctop - 0.035), 0.065, 0.60, ZINC, rot=across)
        legs = [a0 + 0.25 + k * 1.05 for k in range(int((L - 0.5) / 1.05) + 1)]
        for a in legs:                                                                           # legs, foot plates, cross braces
            for sy in (-1, 1):
                box(P(a, sy * 0.29, (ctop - 0.15 + fz) / 2), (0.05, 0.05, ctop - 0.15 - fz), DARKL)
                box(P(a, sy * 0.29, fz + 0.01), (0.14, 0.14, 0.02), (0.30, 0.31, 0.33))
            box(P(a, 0, fz + 0.30), Z(0.04, 0.58, 0.04), DARKL)
        for a_, b_ in zip(legs[:-1], legs[1:]):
            for sy in (-1, 1):
                box(P((a_ + b_) / 2, sy * 0.29, fz + 0.30), Z(b_ - a_, 0.04, 0.04), DARKL)
        if drive:                                                                                # drive motor + gearbox
            box(P(a1 + 0.18, 0.02, ctop - 0.20), Z(0.22, 0.20, 0.18), (0.20, 0.22, 0.25))
            cyl(P(a1 + 0.18, 0.02, ctop - 0.20), 0.07, 0.30, (0.28, 0.30, 0.34), rot=along)
            box(P(a1 - 0.30, -0.36, ctop - 0.18), Z(0.16, 0.06, 0.20), ZINC)                    # control box on the rail
        for a in [a0 + 0.45 + k * 0.85 for k in range(int((L - 0.8) / 0.85) + 1)]:              # guard-rail posts
            for sy in (-1, 1):
                box(P(a, sy * 0.335, ctop + 0.07), (0.025, 0.025, 0.15), ZINC)
        for sy in (-1, 1):                                                                       # top guard rail
            box(P(mid, sy * 0.335, ctop + 0.14), Z(L - 0.5, 0.02, 0.02), ZINC)
        for a, kind in items:                                                                    # things riding the belt
            if kind == "tote":
                box(P(a, 0, ctop + 0.115), Z(0.55, 0.42, 0.22), BINGREY)
                box(P(a, 0, ctop + 0.23), Z(0.58, 0.45, 0.025), (0.34, 0.36, 0.38))
            else:
                box(P(a, 0, ctop + 0.13), Z(0.40, 0.34, 0.26), CARDBOARD)
                box(P(a, 0, ctop + 0.262), Z(0.10, 0.34, 0.004), (0.78, 0.70, 0.52))             # packing tape

    conveyor_leg("x", cy0, cx_a, cx_end, rollers=(True, False), drive=True,
                 items=((-2.50, "tote"), (-1.35, "carton"), (-0.10, "tote"), (1.00, "carton"), (2.05, "tote"), (2.90, "carton"),
                        (3.95, "tote"), (5.00, "carton")))
    # yellow safety rail in front of the conveyor, the same length (posts + two rails)
    RAIL_Y = cy0 - 2.15
    for xx in [cx_a + 0.2 + k * 1.1 for k in range(int((cx_end - cx_a - 0.4) / 1.1) + 1)]:
        box((xx, RAIL_Y, fz + 0.55), (0.07, 0.07, 1.10), YELLOW)
    for zz in (0.60, 1.00):
        box(((cx_a + cx_end) / 2, RAIL_Y, fz + zz), (cx_end - cx_a - 0.2, 0.04, 0.05), YELLOW)

    # ── yellow industrial robot: a pick station in front of the conveyor, arm reaching over the belt ───────────────────
    bx_, by_ = 0.20, cy0 - 1.25                                  # base position (clear of the wall, inside the safety rail)
    DARKY = (0.14, 0.15, 0.17)
    def along(theta_deg, length):                                # unit direction in the arm's y-z plane, theta from vertical
        t = math.radians(theta_deg)
        return np.array([0.0, math.sin(t), math.cos(t)]) * length
    def link_box(p_from, theta_deg, length, w, d, color):        # box whose long axis runs from p_from along theta
        c = np.array(p_from) + along(theta_deg, length) / 2.0
        box(tuple(c), (w, d, length), color, rot=(-theta_deg, 0, 0))
    def link_cyl(p_from, theta_deg, length, r, color):
        c = np.array(p_from) + along(theta_deg, length) / 2.0
        cyl(tuple(c), r, length, color, rot=(-theta_deg, 0, 0))
    box((bx_, by_, fz + 0.02), (0.80, 0.80, 0.04), (0.30, 0.31, 0.33))                          # floor plate
    box((bx_, by_, fz + 0.17), (0.54, 0.54, 0.26), DARKY)                                       # pedestal
    for ox in (-0.34, 0.34):
        for oy in (-0.34, 0.34):
            cyl((bx_ + ox, by_ + oy, fz + 0.06), 0.03, 0.04, (0.55, 0.57, 0.60))               # anchor bolts
    cyl((bx_, by_, fz + 0.42), 0.25, 0.24, YELLOW)                                              # J1 rotating base
    cyl((bx_, by_, fz + 0.545), 0.255, 0.02, DARKY)                                             # seal ring
    box((bx_, by_ - 0.04, fz + 0.76), (0.40, 0.42, 0.46), YELLOW)                               # shoulder housing
    for sx in (-1, 1):
        cyl((bx_ + sx * 0.21, by_ - 0.04, fz + 0.88), 0.14, 0.04, DARKY, rot=(0, 90, 0))        # J2 motor caps
    sh = np.array([bx_, by_ - 0.04, fz + 0.88])                                                 # shoulder pivot
    th1, th2 = 22.0, 100.0
    link_box(sh, th1, 0.95, 0.24, 0.26, YELLOW)                                                 # upper arm
    e1 = sh + along(th1, 0.95)
    link_box(sh + along(th1, 0.10), th1, 0.20, 0.27, 0.29, YELLOW)                              # arm root bulge
    cyl(tuple(e1), 0.15, 0.34, YELLOW, rot=(0, 90, 0))                                          # J3 elbow housing
    for sx in (-1, 1):
        cyl((e1[0] + sx * 0.18, e1[1], e1[2]), 0.11, 0.03, DARKY, rot=(0, 90, 0))
    link_box(e1, th2, 0.70, 0.17, 0.19, YELLOW)                                                 # forearm
    e2 = e1 + along(th2, 0.70)
    link_cyl(e2 - along(th2, 0.08), th2, 0.20, 0.085, YELLOW)                                   # J4 forearm roll
    link_cyl(e2 + along(th2, 0.12), th2, 0.10, 0.075, DARKY)                                    # J5 wrist joint
    w0 = e2 + along(th2, 0.22)
    cyl(tuple(w0 + np.array([0, 0, -0.07])), 0.07, 0.16, YELLOW)                                # J6 wrist, turned downward
    cyl(tuple(w0 + np.array([0, 0, -0.18])), 0.055, 0.06, DARKY)                                # tool flange
    cyl(tuple(w0 + np.array([0, 0, -0.26])), 0.075, 0.05, BLACK)                                # suction cup
    for k in range(3):                                                                          # umbilical cable along the arm
        pk = sh + along(th1, 0.15 + k * 0.30) + np.array([0.13, 0.0, 0.0])
        cyl(tuple(pk), 0.016, 0.34, BLACK, rot=(-th1, 0, 0))

    # ── signal tower (red / amber / green) on a pole, near the right wall ───────────────────────────────────────────
    tx, ty = 2.60, RIGHT_Y - 0.35
    cyl((tx, ty, fz + 0.75), 0.025, 1.50, DARKGREY)
    cyl((tx, ty, fz + 1.58), 0.06, 0.12, (0.72, 0.10, 0.10))
    cyl((tx, ty, fz + 1.70), 0.06, 0.12, (0.85, 0.62, 0.05))
    cyl((tx, ty, fz + 1.82), 0.06, 0.12, (0.10, 0.62, 0.20))
    box((tx, ty, fz + 0.04), (0.20, 0.20, 0.08), DARKGREY)

    print(f"[Factory background] built {counter[0]} visual prims under {ROOT}.", flush=True)
