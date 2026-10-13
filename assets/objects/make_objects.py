"""Generate graspable STL objects (meters, centered at origin, ~5 cm, fits Panda gripper)."""
import os
import numpy as np

OUT = os.path.dirname(os.path.abspath(__file__))


def write_stl(name, tris):
    tris = np.asarray(tris, dtype=np.float64)
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
    # volume (signed) as an outward-winding sanity check
    vol = np.einsum("ij,ij->i", tris[:, 0], np.cross(tris[:, 1], tris[:, 2])).sum() / 6
    print(f"{name}.stl  {len(tris)} tris  volume={vol*1e6:.1f} cm^3")


def prism(poly, h):
    """Extrude a CCW 2D polygon (convex) along z, centered."""
    poly = np.asarray(poly)
    n = len(poly)
    lo = np.c_[poly, np.full(n, -h / 2)]
    hi = np.c_[poly, np.full(n, h / 2)]
    tris = []
    for i in range(1, n - 1):
        tris.append([hi[0], hi[i], hi[i + 1]])
        tris.append([lo[0], lo[i + 1], lo[i]])
    for i in range(n):
        j = (i + 1) % n
        tris.append([lo[i], lo[j], hi[j]])
        tris.append([lo[i], hi[j], hi[i]])
    return tris


def ngon(n, r, rot=0.0):
    a = rot + np.arange(n) * 2 * np.pi / n
    return np.c_[r * np.cos(a), r * np.sin(a)]


def cone(r, h, seg=48):
    ring = ngon(seg, r)
    base_z, apex = -h / 2, np.array([0, 0, h / 2])
    c = np.array([0, 0, base_z])
    tris = []
    for i in range(seg):
        j = (i + 1) % seg
        a, b = np.r_[ring[i], base_z], np.r_[ring[j], base_z]
        tris.append([c, b, a])
        tris.append([a, b, apex])
    return tris


def sphere(r, nu=32, nv=16):
    def p(u, v):
        return r * np.array([np.sin(v) * np.cos(u), np.sin(v) * np.sin(u), np.cos(v)])
    tris = []
    for i in range(nu):
        u0, u1 = 2 * np.pi * i / nu, 2 * np.pi * (i + 1) / nu
        for k in range(nv):
            v0, v1 = np.pi * k / nv, np.pi * (k + 1) / nv
            a, b, c, d = p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)
            if k > 0:
                tris.append([a, c, b])
            if k < nv - 1:
                tris.append([a, d, c])
    return tris


def torus_free_ring(R, r, nu=48, nv=16):
    def p(u, v):
        return np.array([(R + r * np.cos(v)) * np.cos(u), (R + r * np.cos(v)) * np.sin(u), r * np.sin(v)])
    tris = []
    for i in range(nu):
        u0, u1 = 2 * np.pi * i / nu, 2 * np.pi * (i + 1) / nu
        for k in range(nv):
            v0, v1 = 2 * np.pi * k / nv, 2 * np.pi * (k + 1) / nv
            a, b, c, d = p(u0, v0), p(u1, v0), p(u1, v1), p(u0, v1)
            tris += [[a, b, c], [a, c, d]]
    return tris


def l_block(s=0.05, t=0.025, h=0.04):
    pts = [(-s/2, -s/2), (s/2, -s/2), (s/2, -s/2 + t), (-s/2 + t, -s/2 + t), (-s/2 + t, s/2), (-s/2, s/2)]
    # concave: split into two convex boxes
    box1 = prism([(-s/2, -s/2), (s/2, -s/2), (s/2, -s/2 + t), (-s/2, -s/2 + t)], h)
    box2 = prism([(-s/2, -s/2 + t), (-s/2 + t, -s/2 + t), (-s/2 + t, s/2), (-s/2, s/2)], h)
    return box1 + box2  # touching boxes; fine for collision/visual


write_stl("cylinder", prism(ngon(48, 0.0225), 0.05))
write_stl("sphere", sphere(0.025))
write_stl("cone", cone(0.0275, 0.06))
write_stl("hex_prism", prism(ngon(6, 0.03), 0.045))
write_stl("triangular_prism", prism(ngon(3, 0.035, np.pi / 2), 0.05))
write_stl("cuboid_bar", prism([(-0.04, -0.0175), (0.04, -0.0175), (0.04, 0.0175), (-0.04, 0.0175)], 0.035))
write_stl("ring", torus_free_ring(0.025, 0.01))
write_stl("l_block", l_block())
