"""Florence flask + beaker STLs (meters). Origin = center of the flat base, +Z up.

Body: 5 cm sphere (flat-bottomed so it stands). Neck: 2 cm outer diameter, 2 mm glass wall.
  flask_glass.stl   hollow glass shell  (use for all three flasks)
  flask_liquid.stl  liquid inside the shell (use for the blue and red flasks; none for the empty one)
  beaker_glass.stl / beaker_liquid.stl  5 cm dia x 7 cm tall beaker (+ liquid, filled to 4.5 cm)
Grasp heights above the base: neck ~ z 5.5-9.0 cm, body center z = 2.29 cm.
"""
import os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))
R, WALL = 0.025, 0.002              # body outer radius, glass thickness
RN = 0.010                          # neck outer radius (2 cm diameter)
RB = 0.010                          # flat base radius
H, LIP_Z, LIP_R = 0.095, 0.093, 0.0115
FILL_Z = 0.034                      # liquid surface height
SEG = 64

zc = np.sqrt(R**2 - RB**2)          # sphere center height so the base circle is r=RB at z=0
Ri, RNi = R - WALL, RN - WALL


def arc(rad, r_from, r_to, n=40, eps=0.0):
    """Points on sphere (center z=zc) from radius r_from to r_to along the lower->upper meridian."""
    def ang(r, lower):
        s = np.arcsin(np.clip((np.sqrt(max(rad**2 - r**2, 0)) * (-1 if lower else 1)) / rad, -1, 1))
        return s
    t0, t1 = ang(r_from, True), ang(r_to, False)
    t = np.linspace(t0, t1, n)
    return [(rad * np.cos(a) - eps, zc + rad * np.sin(a) - eps * np.sin(a)) for a in t]


def arc_between_z(rad, z0, z1, n=40):
    t = np.linspace(np.arcsin((z0 - zc) / rad), np.arcsin((z1 - zc) / rad), n)
    return [(rad * np.cos(a), zc + rad * np.sin(a)) for a in t]


def revolve(profile):
    prof = np.asarray(profile, float)
    th = np.linspace(0, 2 * np.pi, SEG, endpoint=False)
    tris = []
    for (r0, z0), (r1, z1) in zip(prof, np.roll(prof, -1, axis=0)):
        for i in range(SEG):
            j = (i + 1) % SEG
            a, b = (r0*np.cos(th[i]), r0*np.sin(th[i]), z0), (r0*np.cos(th[j]), r0*np.sin(th[j]), z0)
            c, d = (r1*np.cos(th[j]), r1*np.sin(th[j]), z1), (r1*np.cos(th[i]), r1*np.sin(th[i]), z1)
            for t in ([a, b, c], [a, c, d]):
                t = np.array(t)
                if np.linalg.norm(np.cross(t[1]-t[0], t[2]-t[0])) > 1e-14:
                    tris.append(t)
    tris = np.array(tris)
    if volume(tris) < 0:
        tris = tris[:, ::-1]
    return tris


def volume(tris):
    return np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6


def write_stl(name, tris):
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    with open(os.path.join(OUT, name + ".stl"), "w") as f:
        f.write(f"solid {name}\n")
        for t, nn in zip(tris, n):
            f.write(f"facet normal {nn[0]:.6e} {nn[1]:.6e} {nn[2]:.6e}\n outer loop\n")
            for v in t:
                f.write(f"  vertex {v[0]:.6e} {v[1]:.6e} {v[2]:.6e}\n")
            f.write(" endloop\nendfacet\n")
        f.write(f"endsolid {name}\n")
    lo, hi = tris.reshape(-1, 3).min(0) * 100, tris.reshape(-1, 3).max(0) * 100
    print(f"{name}.stl {len(tris)} tris  vol={volume(tris)*1e6:.1f} cm^3  "
          f"bbox cm x[{lo[0]:.2f},{hi[0]:.2f}] z[{lo[2]:.2f},{hi[2]:.2f}]")


# ---- glass shell: closed (r,z) loop, revolved ----
outer = [(0, 0)] + arc(R, RB, RN) + [(RN, LIP_Z - 0.003), (LIP_R, LIP_Z - 0.003), (LIP_R, H), (RNi, H)]
z_bot = WALL
t_in = arc_between_z(Ri, z_bot, zc + np.sqrt(Ri**2 - RNi**2))  # lower -> neck junction
inner = t_in[::-1]
glass = outer + inner + [(0, z_bot)]
write_stl("flask_glass", revolve(glass))

# ---- liquid: cavity up to FILL_Z (shrunk 0.2 mm so it doesn't z-fight the glass) ----
Rl = Ri - 0.0002
zl0 = z_bot + 0.0002
liq = [(0, zl0)] + arc_between_z(Rl, zl0, FILL_Z) + [(0, FILL_Z)]
write_stl("flask_liquid", revolve(liq))

# ---- Beaker: 5 cm diameter, 7 cm tall, open top, 2 mm wall, 3 mm floor. Origin = base centre. ----
BR, BH, BWALL, BFLOOR, BFILL = 0.025, 0.07, 0.002, 0.003, 0.045
beaker = [(0, 0), (BR, 0), (BR, BH), (BR - BWALL, BH), (BR - BWALL, BFLOOR), (0, BFLOOR)]
write_stl("beaker_glass", revolve(beaker))
beaker_liq = [(0, BFLOOR + 0.0002), (BR - BWALL - 0.0002, BFLOOR + 0.0002),
              (BR - BWALL - 0.0002, BFILL), (0, BFILL)]
write_stl("beaker_liquid", revolve(beaker_liq))
