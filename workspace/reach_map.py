"""Reachability map for the constant-pitch sideways hand.

The hand orientation is fixed (approach pointing radially outward from the robot base, pitched down by
side_grasp.HAND_PITCH_DEG, finger-closing axis horizontal). With that orientation locked, a 3-D position grid in
polar coordinates (radius r, angle theta, height z) is flood-filled with IK solutions: starting from one known
solution, every neighbouring cell is solved with the neighbour's configuration as warm start, so every stored
configuration is joint-continuous with its neighbours. Cells that cannot be reached continuously stay empty.

The planner then finds carries by searching this grid (Dijkstra), instead of assuming that an arc or a straight
line happens to be feasible. The map also answers "can the arm place here?" for any shelf slot directly.

Build (once, a few minutes):  python reach_map.py   (run inside the Isaac Sim container, see build_cli)
"""
import os
import sys
import time
import heapq
from collections import deque
import numpy as np

R_AXIS = np.arange(0.28, 0.90, 0.025)
T_AXIS = np.deg2rad(np.arange(-70.0, 150.1, 7.5))
Z_AXIS = np.arange(0.05, 0.86, 0.03)
MAP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reach_map_p{pitch:.0f}_r{roll:.0f}.npz")


def cell_pos(ir, it, iz):
    r, t, z = R_AXIS[ir], T_AXIS[it], Z_AXIS[iz]
    return np.array([r * np.cos(t), r * np.sin(t), z])


def cell_of(pos):
    r, t = np.hypot(pos[0], pos[1]), np.arctan2(pos[1], pos[0])
    ir = int(np.clip(round((r - R_AXIS[0]) / (R_AXIS[1] - R_AXIS[0])), 0, len(R_AXIS) - 1))
    it = int(np.clip(round((t - T_AXIS[0]) / (T_AXIS[1] - T_AXIS[0])), 0, len(T_AXIS) - 1))
    iz = int(np.clip(round((pos[2] - Z_AXIS[0]) / (Z_AXIS[1] - Z_AXIS[0])), 0, len(Z_AXIS) - 1))
    return ir, it, iz


class ReachMap:
    def __init__(self, Q, pitch, roll=0.0):
        self.Q = Q                 # (nr, nt, nz, 7), NaN where unreachable
        self.pitch = pitch
        self.roll = roll

    @property
    def ok(self):
        return ~np.isnan(self.Q[..., 0])

    # ---- build -------------------------------------------------------------------------------------
    @classmethod
    def build(cls, planner, seed_q, seed_pos, log=print):
        import side_grasp as sg
        nr, nt, nz = len(R_AXIS), len(T_AXIS), len(Z_AXIS)
        Q = np.full((nr, nt, nz, 7), np.nan)
        t0 = time.time()
        c0 = cell_of(seed_pos)
        a0 = np.array([np.cos(T_AXIS[c0[1]]), np.sin(T_AXIS[c0[1]])])
        q0 = planner._ik_a(cell_pos(*c0), a0, planner.roll, seed_q)
        if q0 is None:
            raise RuntimeError("seed cell has no IK solution")
        Q[c0] = q0
        dq = deque([c0])
        done = 0
        while dq:
            c = dq.popleft()
            for d in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                n = (c[0] + d[0], c[1] + d[1], c[2] + d[2])
                if not (0 <= n[0] < nr and 0 <= n[1] < nt and 0 <= n[2] < nz) or not np.isnan(Q[n][0]):
                    continue
                a = np.array([np.cos(T_AXIS[n[1]]), np.sin(T_AXIS[n[1]])])
                best, best_d = None, 1e9
                for roll in (planner.roll,):
                    cand = planner._ik_a(cell_pos(*n), a, roll, Q[c])
                    if cand is not None:
                        dd = float(np.max(np.abs(cand - Q[c])))
                        if dd < best_d:
                            best, best_d = cand, dd
                if best is not None and best_d < 0.45:
                    Q[n] = best
                    dq.append(n)
            done += 1
            if done % 500 == 0:
                log(f"  reach map: {done} cells expanded, {int((~np.isnan(Q[..., 0])).sum())} reachable, {time.time() - t0:.0f}s")
        log(f"reach map built: {int((~np.isnan(Q[..., 0])).sum())}/{Q[..., 0].size} cells reachable in {time.time() - t0:.0f}s")
        return cls(Q, sg.HAND_PITCH_DEG, planner.roll)

    def save(self):
        np.savez_compressed(MAP_PATH.format(pitch=self.pitch, roll=np.degrees(self.roll)), Q=self.Q.astype(np.float32),
                            pitch=self.pitch, roll=self.roll)

    @classmethod
    def load(cls, pitch, roll=0.0):
        path = MAP_PATH.format(pitch=pitch, roll=np.degrees(roll))
        if not os.path.exists(path):
            return None
        d = np.load(path)
        return cls(d["Q"].astype(float), float(d["pitch"]), roll)

    # ---- queries -----------------------------------------------------------------------------------
    def nearest_reachable(self, pos, max_cells=2):
        c0 = cell_of(pos)
        best, best_d = None, 1e9
        for dr in range(-max_cells, max_cells + 1):
            for dt in range(-max_cells, max_cells + 1):
                for dz in range(-max_cells, max_cells + 1):
                    c = (c0[0] + dr, c0[1] + dt, c0[2] + dz)
                    if 0 <= c[0] < len(R_AXIS) and 0 <= c[1] < len(T_AXIS) and 0 <= c[2] < len(Z_AXIS) \
                            and not np.isnan(self.Q[c][0]):
                        d = np.linalg.norm(cell_pos(*c) - pos)
                        if d < best_d:
                            best, best_d = c, d
        return best

    def path(self, start_pos, goal_pos, blocked=lambda c, q: False, budget_s=40.0, extra_cost=lambda c: 0.0):
        """A* over reachable cells from start_pos to goal_pos (weighted 1.3 towards the goal: far fewer cells expanded
        than plain Dijkstra, which flooded the whole map); returns a list of positions (cell centres)."""
        s, g = self.nearest_reachable(start_pos), self.nearest_reachable(goal_pos)
        if s is None or g is None:
            return None
        dist = {s: 0.0}
        prev = {}
        t_end = time.time() + budget_s       # an unreachable goal would otherwise explore the whole map
        goal_xyz = cell_pos(*g)

        def h(c):
            return 1.3 * float(np.linalg.norm(cell_pos(*c) - goal_xyz))

        pq = [(h(s), 0.0, s)]
        memo = {}
        cost_memo = {}

        def is_blocked(c):
            if c not in memo:
                memo[c] = blocked(c, self.Q[c])
            return memo[c]

        while pq:
            if time.time() > t_end:
                return None
            _f, d, c = heapq.heappop(pq)
            if c == g:
                out = [c]
                while out[-1] in prev:
                    out.append(prev[out[-1]])
                return [cell_pos(*x) for x in reversed(out)]
            if d > dist.get(c, 1e9):
                continue
            for dd in ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)):
                n = (c[0] + dd[0], c[1] + dd[1], c[2] + dd[2])
                if not (0 <= n[0] < len(R_AXIS) and 0 <= n[1] < len(T_AXIS) and 0 <= n[2] < len(Z_AXIS)):
                    continue
                if np.isnan(self.Q[n][0]) or is_blocked(n):
                    continue
                jump = float(np.max(np.abs(self.Q[n] - self.Q[c])))
                if jump > 0.5:
                    continue
                if n not in cost_memo:               # setdefault(n, extra_cost(n)) would recompute it every time
                    cost_memo[n] = extra_cost(n)
                nd = d + float(np.linalg.norm(cell_pos(*n) - cell_pos(*c))) * (1.0 + 2.0 * jump) + cost_memo[n] * 0.02
                if nd < dist.get(n, 1e9):
                    dist[n] = nd
                    prev[n] = c
                    heapq.heappush(pq, (nd + h(n), nd, n))
        return None


def build_cli():
    """Run inside the Isaac Sim container: builds and saves the map for side_grasp.HAND_PITCH_DEG."""
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from isaacsim import SimulationApp
    app = SimulationApp({"headless": True})
    from isaacsim.robot_motion.motion_generation import interface_config_loader as L
    from isaacsim.robot_motion.motion_generation.lula import LulaKinematicsSolver
    import side_grasp as sg
    cfg = L.load_supported_motion_policy_config("Franka", "RMPflow")
    ks = LulaKinematicsSolver(robot_description_path=cfg["robot_description_path"], urdf_path=cfg["urdf_path"])
    P = sg.SideGraspPlanner(ks)
    seed_pos = np.array([0.55, 0.0, 0.135])
    for roll in sg.ROLLS:
        P.set_roll(roll)
        seed = P.solve_near(seed_pos, np.array([1.0, 0.0]), sg.HOME_Q)
        if seed is None:
            print("REACH roll %.0f: no seed solution" % np.degrees(roll), flush=True)
            continue
        m = ReachMap.build(P, seed, seed_pos)
        m.save()
        ok = m.ok
        print("REACH roll %.0f deg: %d/%d cells reachable" % (np.degrees(roll), int(ok.sum()), ok.size), flush=True)
        for iz, z in enumerate(Z_AXIS):
            if iz % 3 == 0:
                print("REACH   z=%.2f  %.2f" % (z, ok[:, :, iz].mean()), flush=True)
    app.close()


if __name__ == "__main__":
    build_cli()
