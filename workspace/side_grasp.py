"""Sideways (horizontal) grasp planner for flasks and beakers.

The hand points horizontally, along the line from the robot base to the vessel, and the fingers close sideways on
the neck (flask) or the upper body (beaker). Everything is planned up front in joint space with Lula IK:

  * every key pose is an IK solution nearest the previous configuration;
  * every motion that happens WHILE HOLDING a vessel (lift, pull back, carry, insert, retreat) is a chain of
    1 cm Cartesian steps with the hand orientation locked level (IK orientation tolerance ~3 deg), so the vessel
    stays upright and liquid cannot spill. The carry from table to shelf is an arc around the robot base at
    constant height, with the hand always pointing radially outward. Joint-space interpolation is only used for
    the empty arm (home -> pre-grasp, final return home);
  * every sample of every segment is checked against the shelf (planks, posts) and the table with capsule-style
    points along the arm links; plans that would hit anything are rejected, not played.

No Isaac dependency of its own: it needs an object with Lula's compute_inverse_kinematics /
compute_forward_kinematics, so the same code is tested offline.
"""
from dataclasses import dataclass
import numpy as np

FRAME = "right_gripper"
HOME_Q = np.array([0.00, -0.785, 0.00, -2.356, 0.00, 1.571, 0.785])

GRASP_HEIGHT = {"flask": 0.075, "beaker": 0.052}   # finger height above the vessel's base centre
PRE_STANDOFF = 0.12
MIN_PRE_RADIUS = 0.27
SET_DOWN_CLEARANCE = 0.006
CART_STEP = 0.01
MAX_JUMP = 0.20          # rad between two 1 cm Cartesian steps (rejects IK branch flips)
MAX_TILT_DEG = 5.0       # allowed tilt of the held vessel's axis from vertical

# default shelf (overridden by set_shelf)
# Wall-mounted unit on the robot's +Y side: planks x0..x1 by y0..y1, a back panel at y1, open toward the robot (-Y).
# Stepped like a tiered display: each higher tier starts further back (y0s), so no plank hangs over the slot below it,
# where the pitched forearm needs the room. Slots sit SLOT_INSET behind each tier's front edge.
SHELF_DEFAULT = dict(tops=[0.08, 0.36, 0.64], x0=-0.05, x1=0.65, y0s=[0.44, 0.53, 0.62], y1=0.86,
                     plank_t=0.02, back_t=0.03)
SLOT_INSET = 0.075


class PlanError(RuntimeError):
    pass


@dataclass
class Segment:
    name: str
    q: np.ndarray                 # (N, 7) joint waypoints
    gripper: str = None           # "open" / "close" once the segment has arrived
    fast: bool = True


def _quat_from_matrix(m):
    from isaacsim.core.utils.numpy.rotations import rot_matrices_to_quats
    return rot_matrices_to_quats(m)


ROLLS = (0.0, np.pi)
HAND_PITCH_DEG = 40.0     # hand approach pitched down by this much; constant through a carry so the vessel stays upright


def hand_rot(approach_xy, roll=0.0, pitch_deg=None):
    """Rotation matrix (columns = tool x, y, z). Tool z = horizontal approach pitched DOWN by pitch_deg; tool y
    (finger closing axis) stays horizontal; tool x is then ~down (vessel axis)."""
    pitch = np.radians(HAND_PITCH_DEG if pitch_deg is None else pitch_deg)
    a = np.array([approach_xy[0], approach_xy[1], 0.0], float)
    a /= np.linalg.norm(a)
    z = np.array([a[0] * np.cos(pitch), a[1] * np.cos(pitch), -np.sin(pitch)])
    y0 = np.array([-a[1], a[0], 0.0])
    x0 = np.cross(y0, z)
    c, s_ = np.cos(roll), np.sin(roll)
    x = c * x0 + s_ * y0
    y = np.cross(z, x)
    return np.stack([x, y, z], axis=1)


def hand_quat(approach_xy, roll=0.0, pitch_deg=None):
    return _quat_from_matrix(hand_rot(approach_xy, roll, pitch_deg))


def _rot_err_deg(R_a, R_b):
    c = (float(np.trace(R_a.T @ R_b)) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0))))


def _unit2(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


class SideGraspPlanner:
    ARM_FRAMES = ("panda_link2", "panda_link3", "panda_link4", "panda_link5", "panda_link6", "panda_link7")
    ARM_R = 0.09              # link radius incl. safety margin

    def __init__(self, kin, frame=FRAME):
        self.kin = kin
        self.frame = frame
        self.rng = np.random.default_rng(7)
        self.set_shelf(**SHELF_DEFAULT)
        self.vessel_cols = []        # [(name, base xyz, height)] vessels the arm must not touch (all but the target)
        # The gripper is symmetric, so every hand pose has two IK families: roll 0 and roll pi (hand turned 180 deg about
        # its own axis). They are physically different arm configurations with different reach, and the roll must NOT
        # change while a vessel is held (that would turn it over). So one reach map per roll, and a plan uses one roll.
        self._reach_maps = {}
        self._cell_caches = {r: {} for r in ROLLS}   # reach-map cell -> (arm points, grasp point): static, lazy
        try:
            from reach_map import ReachMap
            for r in ROLLS:
                self._reach_maps[r] = ReachMap.load(HAND_PITCH_DEG, r)
        except Exception:
            self._reach_maps = {}
        self.roll = ROLLS[0]
        self.reach = None
        self._cell_cache = self._cell_caches[ROLLS[0]]
        self.set_roll(ROLLS[0])

    def set_roll(self, roll):
        self.roll = roll
        self.reach = self._reach_maps.get(roll)
        self._cell_cache = self._cell_caches[roll]

    # ---- environment -----------------------------------------------------------------------------
    def set_shelf(self, tops, x0, x1, y0s, y1, plank_t=0.02, back_t=0.03):
        self.rack_tops = list(tops)
        self.shelf = dict(x0=x0, x1=x1, y0s=list(y0s), y1=y1)
        boxes = []                                  # (min xyz, max xyz)
        for t, y0 in zip(tops, y0s):
            boxes.append((np.array([x0, y0, t - plank_t]), np.array([x1, y1, t])))
        zmax = tops[-1] + 0.25
        boxes.append((np.array([x0, y1, -0.05]), np.array([x1, y1 + back_t, zmax])))        # back panel
        self.boxes = boxes

    def set_vessels(self, vessels):
        """vessels: [(name, base_centre_xyz)] of every vessel except the one being moved."""
        self.vessel_cols = [(n, np.asarray(p, float), 0.10) for n, p in vessels]

    def slot_y(self, tier):
        return self.shelf["y0s"][tier] + SLOT_INSET

    def _hits_box(self, p, r, above_ok=False, rz=None):
        for lo, hi in self.boxes:
            if above_ok and p[2] >= hi[2] and (hi[2] - lo[2]) < 0.05:
                continue          # hand/vessel region resting above a plank: the vessel-bottom check covers it
            d = np.maximum(np.maximum(lo - p, p - hi), 0.0)
            if rz is not None:
                d = d * np.array([1.0, 1.0, r / rz])         # the hand is wide but thin: tighter vertical extent
            if float(d @ d) < r * r:
                return True
        return False

    # ---- kinematics helpers ------------------------------------------------------------------------
    def _fk(self, q, frame=None):
        pos, rot = self.kin.compute_forward_kinematics(frame or self.frame, q)
        return np.asarray(pos, float), np.asarray(rot, float)

    def hand_pose(self, q):
        p, rot = self._fk(q)
        return p, _unit2(rot[:2, 2])

    def tilt_deg(self, q):
        """How far the hand is from the intended orientation (same roll, level fingers, constant pitch), in degrees.
        Anything above a few degrees means the carried vessel tilts - or, near 180, has been turned over."""
        _, rot = self._fk(q)
        return _rot_err_deg(hand_rot(_unit2(rot[:2, 2]), self.roll), rot)

    def _ik_raw(self, pos, quat, seed, pos_tol, ori_tol):
        q, ok = self.kin.compute_inverse_kinematics(self.frame, np.asarray(pos, float), quat, warm_start=seed,
                                                    position_tolerance=pos_tol, orientation_tolerance=ori_tol)
        return np.asarray(q, float) if ok else None

    def _ik_once(self, pos, quat, seed, pos_tol=0.003, ori_tol=0.03):
        """Two-stage IK: a loose-orientation solve honours the warm start (Lula jumps to far branches when the
        tolerance is tight), then a tight solve refines it. Accepted only if the hand really is level and aimed."""
        q = self._ik_raw(pos, quat, seed, pos_tol, 0.15)
        if q is None:
            return None
        q2 = self._ik_raw(pos, quat, q, pos_tol, ori_tol)
        if q2 is not None and float(np.max(np.abs(q2 - q))) < 0.3:
            q = q2
        p2, rot = self._fk(q)
        if np.linalg.norm(p2 - pos) > pos_tol * 1.5:
            return None
        return q if self._orient_ok(rot) else None

    def _ik_a(self, pos, a, roll, seed, pos_tol=0.003, ori_tol=0.03):
        """Like _ik_once, but the result must match the exact target rotation (same roll, within 4 deg): a solution on the
        other roll family, or a tilted hand, is rejected."""
        R_t = hand_rot(a, roll)
        quat = _quat_from_matrix(R_t)
        q = self._ik_raw(pos, quat, seed, pos_tol, 0.15)
        if q is None:
            return None
        q2 = self._ik_raw(pos, quat, q, pos_tol, ori_tol)
        if q2 is not None and float(np.max(np.abs(q2 - q))) < 0.3:
            q = q2
        p2, rot = self._fk(q)
        if np.linalg.norm(p2 - pos) > pos_tol * 1.5:
            return None
        return q if _rot_err_deg(R_t, rot) < 4.0 else None

    def _orient_ok(self, rot):
        """Finger-closing axis horizontal and approach pitched as commanded (within ~3.5 deg)."""
        pitch = np.radians(HAND_PITCH_DEG)
        return abs(rot[2, 1]) < 0.08 and abs(rot[2, 2] + np.sin(pitch)) < 0.08

    def candidates(self, pos, approach_xy, q_ref, tries=40):
        found = []
        for roll in (self.roll,):
            for k in range(tries):
                seed = q_ref if k == 0 else q_ref + self.rng.normal(0.0, 0.3 + 0.03 * k, 7)
                q = self._ik_a(pos, approach_xy, roll, seed)
                if q is not None and all(np.max(np.abs(q - f)) > 0.05 for f in found):
                    found.append(q)
        found.sort(key=lambda q: float(np.max(np.abs(q - q_ref))))
        return found

    def solve_near(self, pos, approach_xy, q_ref):
        c = self.candidates(pos, approach_xy, q_ref, tries=14)
        return c[0] if c else None

    # ---- collision checks --------------------------------------------------------------------------
    def arm_points(self, q, holding):
        """(point, radius) pairs along the arm; the hand region is included only while it must stay clear."""
        pts = []
        prev = None
        for name in self.ARM_FRAMES:
            try:
                p, _ = self._fk(q, name)
            except Exception:
                continue
            if prev is not None:
                n = max(1, int(np.linalg.norm(p - prev) / 0.04))
                for i in range(1, n + 1):
                    pts.append((prev + (p - prev) * i / n, self.ARM_R, False, None))
            else:
                pts.append((p, self.ARM_R, False, None))
            prev = p
        if holding is not None:       # hand + fingers + held vessel: spheres around the grasp point and hand
            ee, _ = self._fk(q)
            pts.append((ee, 0.075, True, 0.06))
            try:
                hand, _ = self._fk(q, "panda_hand")
                pts.append((hand, 0.125, True, 0.07))
            except Exception:
                pass
        return pts

    def collides(self, q, holding=0.09, hold_h=0.075, carrying=None, hand_vessels=None, cache=None, rest_ok=False):
        """holding: radius of the hand/grasp-point spheres used for the plank/table checks (None = hand allowed
        between planks, e.g. during the slide-in). carrying: a vessel is in the gripper (checked against every
        other vessel). hand_vessels: the hand body is checked against other vessels. Both default to
        `holding is not None`, but the slide-in passes them explicitly so the vessel and hand are still checked
        against vessels already on the shelf."""
        carrying = (holding is not None) if carrying is None else carrying
        hand_vessels = (holding is not None) if hand_vessels is None else hand_vessels
        # `cache` = (arm_points at holding 0.09, grasp-point position) precomputed for a reach-map cell: the map is
        # static, so its forward kinematics need computing only once, not for every cell every Dijkstra expands.
        plank_pts = cache[0] if cache is not None else self.arm_points(q, holding)
        for p, r, is_hand, rz in plank_pts:
            if np.hypot(p[0], p[1]) > 0.15 and p[2] < r * 0.55 and not (is_hand and p[2] > 0.03):   # table / bench top
                return f"link at z={p[2]:.3f} touches the table"
            if self._hits_box(p, r, above_ok=(rest_ok and is_hand), rz=rz):
                return f"link at {np.round(p, 2).tolist()} hits the shelf"
        vessel_pts = cache[0] if cache is not None else self.arm_points(q, 0.09 if hand_vessels else None)
        for name, base, h in self.vessel_cols:
            for p, r, is_hand, rz in vessel_pts:
                hd = float(np.hypot(p[0] - base[0], p[1] - base[1]))
                dz = max(base[2] - p[2], p[2] - (base[2] + h), 0.0)
                if rz:
                    dz *= r / rz
                if np.hypot(max(hd - 0.03, 0.0), dz) < r:
                    return f"the {'hand' if is_hand else 'arm'} would hit {name}"
            if carrying:   # the vessel being carried against this one
                ee = cache[1] if cache is not None else self._fk(q)[0]
                bottom, top = ee[2] - hold_h, ee[2] - hold_h + 0.10
                if np.hypot(ee[0] - base[0], ee[1] - base[1]) < 0.075 and bottom < base[2] + h and top > base[2]:
                    return f"the carried vessel would hit {name}"
        if holding is not None:       # the carried vessel's WHOLE body must clear every plank and the back panel
            ee = cache[1] if cache is not None else self._fk(q)[0]
            bottom = ee[2] - hold_h
            top = bottom + 0.11
            for lo, hi in self.boxes:
                if (lo[0] - 0.05 < ee[0] < hi[0] + 0.05) and (lo[1] - 0.05 < ee[1] < hi[1] + 0.05) \
                        and hi[2] > bottom + 0.002 and lo[2] < top:
                    return f"the carried vessel would hit the shelf (vessel z {bottom:.3f}-{top:.3f}, plank top {hi[2]:.3f})"
        return None

    _hold_h = 0.075

    def check_segment(self, qs, holding=0.09, tilt=False, stride=1, hold_h=None, carrying=None, hand_vessels=None,
                      rest_ok=False):
        for i in range(0, len(qs), stride):
            why = self.collides(qs[i], holding, hold_h or self._hold_h, carrying, hand_vessels, rest_ok=rest_ok)
            if why:
                raise PlanError(why)
            if tilt and self.tilt_deg(qs[i]) > MAX_TILT_DEG:
                raise PlanError(f"held vessel would tilt {self.tilt_deg(qs[i]):.1f} deg")

    # ---- path builders -----------------------------------------------------------------------------
    def chain(self, samples, q0):
        """IK chain through [(pos, approach_xy), ...] with continuity; returns (N,7). When the warm start fails or
        lands on another branch, the reachability map's stored configuration for that spot is tried; if the arm is
        on a different (but reachable) branch than the map, it is blended onto the map's branch in small joint-space
        steps (the tilt check on the result still guards the vessel)."""
        qs, prev = [], q0
        for p, a in samples:
            best, best_d = None, 1e9
            seeds = [prev] + [prev + self.rng.normal(0.0, 0.04 * k, 7) for k in range(1, 6)]
            map_cand, map_d = None, 1e9
            if self.reach is not None:
                c = self.reach.nearest_reachable(np.asarray(p, float), max_cells=1)
                if c is not None:
                    seeds.append(self.reach.Q[c])
            for k, seed in enumerate(seeds):
                is_map = self.reach is not None and k == len(seeds) - 1 and len(seeds) == 7
                for roll in (self.roll,):
                    cand = self._ik_a(p, a, roll, seed)
                    if cand is not None:
                        d = float(np.max(np.abs(cand - prev)))
                        if is_map and d < map_d:
                            map_cand, map_d = cand, d
                        if d < best_d:
                            best, best_d = cand, d
                if best is not None and best_d <= MAX_JUMP * 0.5:
                    break
            if (best is None or best_d > MAX_JUMP) and map_cand is not None and map_d <= 0.9:
                n = int(np.ceil(map_d / 0.05))
                for t in range(1, n):
                    qs.append(prev + (map_cand - prev) * t / n)
                best, best_d = map_cand, 0.0
            if best is None or best_d > MAX_JUMP:
                raise PlanError(f"no continuous IK at {np.round(p, 3).tolist()}")
            qs.append(best)
            prev = best
        return np.array(qs)

    def line(self, p0, p1, a, q0, step=CART_STEP):
        p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
        n = max(1, int(np.ceil(np.linalg.norm(p1 - p0) / step)))
        return self.chain([(p0 + (p1 - p0) * i / n, a) for i in range(1, n + 1)], q0)

    def arc(self, p0, a0, p1, a1, q0, direction, step=0.015):
        """Level-hand arc around the base from hand position p0 (yaw a0) to p1 (yaw a1), constant height."""
        th0, th1 = np.arctan2(p0[1], p0[0]), np.arctan2(p1[1], p1[0])
        dth = th1 - th0
        dth = dth % (2 * np.pi) if direction > 0 else -((-dth) % (2 * np.pi))
        r0, r1 = np.hypot(p0[0], p0[1]), np.hypot(p1[0], p1[1])
        al0, al1 = np.arctan2(a0[1], a0[0]), np.arctan2(a1[1], a1[0])
        dal = al1 - al0
        dal = dal % (2 * np.pi) if direction > 0 else -((-dal) % (2 * np.pi))
        length = max(abs(dth) * 0.5 * (r0 + r1), abs(r1 - r0), 1e-3)
        n = max(2, int(np.ceil(length / step)))
        samples = []
        for i in range(1, n + 1):
            t = i / n
            th, r, al = th0 + dth * t, r0 + (r1 - r0) * t, al0 + dal * t
            samples.append((np.array([r * np.cos(th), r * np.sin(th), p0[2] + (p1[2] - p0[2]) * t]),
                            np.array([np.cos(al), np.sin(al)])))
        return self.chain(samples, q0)

    def joint_move(self, qa, qb, name, holding=None, fast=True):
        n = max(2, int(np.ceil(np.max(np.abs(qb - qa)) / 0.04)))
        qs = np.array([qa + (qb - qa) * t for t in np.linspace(0, 1, n + 1)[1:]])
        self.check_segment(qs, holding=holding)
        return Segment(name, qs, fast=fast)

    # ---- full motions ------------------------------------------------------------------------------
    @staticmethod
    def approach_vec(a):
        """3-D direction the hand travels when moving onto the vessel (horizontal heading, pitched down)."""
        pr = np.radians(HAND_PITCH_DEG)
        return np.array([a[0] * np.cos(pr), a[1] * np.cos(pr), -np.sin(pr)])

    @staticmethod
    def standoff_for(xy):
        return float(np.clip(np.hypot(xy[0], xy[1]) - MIN_PRE_RADIUS, 0.04, PRE_STANDOFF))

    @staticmethod
    def hand_clear(xy, a, others, so):
        """With the hand pitched down the palm passes above neighbouring vessels; only the fingers (and the
        held vessel) sweep at vessel height: a ~17 cm wide corridor along the approach."""
        a = np.asarray(a, float)
        v_axis = np.array([-a[1], a[0]])
        reach = so * np.cos(np.radians(HAND_PITCH_DEG)) + 0.04
        for o in others:
            d = np.asarray(o, float) - np.asarray(xy, float)
            u = -float(d @ a)
            v = abs(float(d @ v_axis))
            if -0.04 < u < reach and v < 0.085:
                return False
        return True

    def approach_candidates(self, xy, others, so):
        a = _unit2(xy)
        return [a] if self.hand_clear(xy, a, others, so) else []

    def plan_pick(self, kind, obj_xy, obj_z=0.0, q_start=None, others=(), seed=7):
        self.rng = np.random.default_rng(seed)      # deterministic per call; callers retry with other seeds
        q_start = HOME_Q if q_start is None else q_start
        self._hold_h = GRASP_HEIGHT[kind]
        so = self.standoff_for(obj_xy)
        err = PlanError("no hand-clear approach direction")
        for a in self.approach_candidates(obj_xy, others, so):
            try:
                return self._plan_pick_dir(kind, obj_xy, obj_z, q_start, a, so)
            except PlanError as e:
                err = e
        raise err

    def _plan_pick_dir(self, kind, obj_xy, obj_z, q_start, a, so):
        a3 = self.approach_vec(a)
        grasp = np.array([obj_xy[0], obj_xy[1], obj_z + GRASP_HEIGHT[kind]])
        pre = grasp - a3 * so
        hover = pre + np.array([0, 0, 0.10])
        segs = []
        q_pre = self.solve_near(pre, a, q_start)
        down = None
        if q_pre is not None:
            try:
                segs.append(self.joint_move(q_start, q_pre, "move to pre-grasp", holding=None))
                down = np.array([q_pre])
            except PlanError:
                down = None
        if down is None:
            q_hover = self.solve_near(hover, a, q_start)
            if q_hover is None:
                raise PlanError("hover pose unreachable")
            segs.append(self.joint_move(q_start, q_hover, "move above approach"))
            down = self.line(hover, pre, a, q_hover)
            segs.append(Segment("lower to pre-grasp", down))
        if len(down) > 1:
            self.check_segment(down, holding=0.09, hold_h=self._hold_h, rest_ok=True)
        adv = self.line(pre, grasp, a, down[-1])
        self.check_segment(adv, holding=0.09, hold_h=self._hold_h, rest_ok=True)
        segs.append(Segment("advance onto vessel", adv, gripper="close", fast=False))
        lift_p = grasp + np.array([0, 0, 0.06])
        lift = self.line(grasp, lift_p, a, adv[-1])
        self.check_segment(lift, holding=0.09, tilt=True, rest_ok=True)
        segs.append(Segment("lift", lift, fast=False))
        back, back_p = lift, lift_p
        for frac in (1.0, 0.6, 0.3):
            try:
                back_p = lift_p - a3 * so * frac
                back = self.line(lift_p, back_p, a, lift[-1])
                self.check_segment(back, holding=0.09, tilt=True)
                segs.append(Segment("pull back", back))
                break
            except PlanError:
                back, back_p = lift, lift_p
        return segs, back[-1], back_p

    ARC_R = 0.46

    def _carry(self, q_start, dest_pre, a_dest, z_carry):
        if self.reach is not None:
            return self._carry_map(q_start, dest_pre, a_dest)
        return self._carry_arc(q_start, dest_pre, a_dest, z_carry)

    def _plank_cost(self, c):
        """Soft cost that keeps the carried vessel / hand >= ~12 cm from planks and the back panel until the final
        approach, so lifts do not graze the front edges."""
        ee = self._cell_geom(c, self.reach.Q[c])[1]
        cost = 0.0
        for lo, hi in self.boxes:
            d = np.maximum(np.maximum(lo - ee, ee - hi), 0.0)
            dist = float(np.sqrt(d @ d))
            if dist < 0.12:
                cost += (0.12 - dist) * 25.0
        return cost

    def _cell_geom(self, c, q):
        g = self._cell_cache.get(c)
        if g is None:
            g = (self.arm_points(q, 0.09), self._fk(q)[0])
            self._cell_cache[c] = g
        return g

    def _carry_map(self, q_start, dest_pre, a_dest):
        """Level carry found by searching the reachability map (every cell is joint-continuous with its
        neighbours), then refined into 1-cm IK steps and collision/tilt checked."""
        p0, _ = self.hand_pose(q_start)
        blocked = lambda c, q: self.collides(q, 0.09, self._hold_h, cache=self._cell_geom(c, q)) is not None
        cells = self.reach.path(p0, dest_pre, blocked, extra_cost=self._plank_cost)
        if cells is None:
            raise PlanError("no reachable level-hand path to that destination")
        pts = [p0] + cells + [np.asarray(dest_pre, float)]
        samples = []
        for a_pt, b_pt in zip(pts[:-1], pts[1:]):
            n = max(1, int(np.ceil(np.linalg.norm(b_pt - a_pt) / 0.01)))
            for i in range(1, n + 1):
                pt = a_pt + (b_pt - a_pt) * i / n
                samples.append((pt, _unit2(pt[:2])))
        full = self.chain(samples, q_start)
        self.check_segment(full, holding=0.09, tilt=True, hold_h=self._hold_h)
        return [Segment("carry (level)", full)], full[-1]

    def _carry_arc(self, q_start, dest_pre, a_dest, z_carry):
        """Level carry: rise, move out to a safe radius, arc around the base (either way round), come in to the
        pre-place pose. The hand stays level the whole way (checked), so the vessel never tilts."""
        p0, a0 = self.hand_pose(q_start)
        R = max(self.ARC_R, float(np.hypot(p0[0], p0[1])))
        th0 = np.arctan2(p0[1], p0[0])
        mid = np.array([R * np.cos(th0), R * np.sin(th0), p0[2]])            # radially outward first (low)
        out = self.line(p0, mid, a0, q_start) if np.linalg.norm(mid - p0) > 0.005 else np.array([q_start])
        out_pt = np.array([mid[0], mid[1], z_carry])                          # then straight up at that radius
        up = self.line(mid, out_pt, a0, out[-1]) if abs(z_carry - mid[2]) > 0.005 else np.array([out[-1]])
        out = np.vstack([out, up])
        up = out
        thd = np.arctan2(dest_pre[1], dest_pre[0])
        in_pt = np.array([R * np.cos(thd), R * np.sin(thd), z_carry])
        err = PlanError("carry arc failed")
        for direction in ((1, -1) if p0[1] >= 0 else (-1, 1)):
            try:
                arc = self.arc(out_pt, a0, in_pt, _unit2(in_pt[:2]), out[-1], direction)
                cur = arc[-1]
                parts = [out, arc]
                a_in = _unit2(in_pt[:2])
                end = np.array([dest_pre[0], dest_pre[1], z_carry])
                if np.linalg.norm(end - in_pt) > 0.005:
                    seg_in = self.line(in_pt, end, a_in, cur, step=0.015)
                    parts.append(seg_in)
                    cur = seg_in[-1]
                if abs(a_in @ a_dest) < 0.999:           # yaw to the final approach direction (table, skewed)
                    pass
                full = np.vstack(parts)
                self.check_segment(full, holding=0.09, tilt=True, hold_h=self._hold_h)
                if abs(z_carry - dest_pre[2]) > 0.005:
                    tail = self.line(end, dest_pre, a_dest, cur)
                else:
                    tail = np.array([cur])
                self.check_segment(tail, holding=0.09, tilt=True, hold_h=self._hold_h)
                return [Segment("carry (level)", full), Segment("position at destination", tail)], tail[-1]
            except PlanError as e:
                err = e
        raise err

    def plan_place(self, kind, dest_xy, dest_top_z, q_start, others=(), seed=7):
        self.rng = np.random.default_rng(seed)
        shelf = dest_xy[1] > 0.40
        self._hold_h = GRASP_HEIGHT[kind]
        so = PRE_STANDOFF if shelf else self.standoff_for(dest_xy)
        cands = [_unit2(dest_xy)] if shelf else self.approach_candidates(dest_xy, others, so)
        err = PlanError("no hand-clear approach direction")
        for a in cands:
            a3 = self.approach_vec(a)
            grasp = np.array([dest_xy[0], dest_xy[1], dest_top_z + GRASP_HEIGHT[kind] + SET_DOWN_CLEARANCE])
            pre = grasp - a3 * so
            z_carry = max(grasp[2] + 0.02, GRASP_HEIGHT[kind] + 0.17)
            try:
                carry_segs, q_pre = self._carry(q_start, pre, a, z_carry)
                ins = self.line(pre, grasp, a, q_pre)
                self.check_segment(ins, holding=None, tilt=True, carrying=True, hand_vessels=True)   # hand goes between the planks here
                out_p = grasp - a3 * so
                out = self.line(grasp, out_p, a, ins[-1])
                self.check_segment(out, holding=None, tilt=True, carrying=False, hand_vessels=True)
                return carry_segs + [Segment("slide into place", ins, gripper="open", fast=False),
                                     Segment("retreat", out)], out[-1]
            except PlanError as e:
                err = e
        raise err

    def plan_home(self, q_start, seed=7):
        self.rng = np.random.default_rng(seed)
        # Preferred: straight from where the arm is to the home pose (no intermediate "return to front" pose), staying
        # there until the next prompt. The joint-space path is collision-checked against the shelf, table and vessels;
        # only if it would hit something does the longer level detour through the front below get used.
        try:
            return [self.joint_move(q_start, HOME_Q, "return home", holding=0.09)]
        except PlanError:
            pass
        p, a = self.hand_pose(q_start)
        segs, q = [], q_start
        if p[0] < 0.10 or p[1] > 0.40:      # beside / behind the base or at the shelf: come back to the front the way we went (level, empty hand)
            z = max(p[2], 0.25)
            front = np.array([0.36, 0.0, z])
            err = None
            for direction in ((1, -1) if p[1] >= 0 else (-1, 1)):
                try:
                    start = np.array([p[0], p[1], z])
                    up = self.line(p, start, a, q) if abs(z - p[2]) > 0.005 else np.array([q])
                    arc = self.arc(start, a, front, np.array([1.0, 0.0]), up[-1], direction)
                    full = np.vstack([up, arc])
                    self.check_segment(full, holding=0.09)
                    segs.append(Segment("return to front", full))
                    q = arc[-1]
                    err = None
                    break
                except PlanError as e:
                    err = e
            if err is not None:
                raise err
        segs.append(self.joint_move(q, HOME_Q, "return home", holding=0.09))
        return segs
