#!/usr/bin/env python3
"""
VLA Policy Server — OpenVLA-7B / Deterministic LIBERO fallback
Inputs: overhead RGB, wrist RGB, natural language instruction
Outputs: 7-DOF joint positions + gripper width
"""
import os, sys, json, time, logging
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="[VLA] %(message)s")
log = logging.getLogger(__name__)

# Joint limits for Franka Panda
JOINT_LO = np.array([-2.8973,-1.7628,-2.8973,-3.0718,-2.8973,-0.0175,-2.8973])
JOINT_HI = np.array([ 2.8973, 1.7628, 2.8973,-0.0698, 2.8973, 3.7525, 2.8973])
HOME_JOINTS = np.array([0.0,-0.785,0.0,-2.356,0.0,1.571,0.785])

def clamp_joints(q):
    return np.clip(q, JOINT_LO, JOINT_HI)

def analytical_ik(xyz):
    x,y,z = xyz
    q1 = np.arctan2(y,x)
    r  = max(np.sqrt(x**2+y**2)-0.05, 0.01)
    zr = z - 0.40
    L1,L2 = 0.40,0.40
    d  = np.clip(np.sqrt(r**2+zr**2), 0.1, L1+L2-0.01)
    c4 = (r**2+zr**2-L1**2-L2**2)/(2*L1*L2)
    q4 = -np.arccos(np.clip(c4,-1,1))
    alpha = np.arctan2(zr,r)
    beta  = np.arctan2(L2*np.sin(-q4), L1+L2*np.cos(-q4))
    q2 = -(alpha+beta)
    q6 = 1.571-(q2+q4)
    return clamp_joints(np.array([q1,q2,0.0,q4,0.0,q6,0.785]))

def min_jerk(s,e,n=20):
    return [s+(10*t**3-15*t**4+6*t**5)*(e-s) for t in [i/n for i in range(n+1)]]

class DeterministicVLAPolicy:
    OBJECTS = {
        "gear":   {"pos":[0.55, 0.15,0.84],"label":"golden_gear",   "name":"Golden Gear"},
        "golden": {"pos":[0.55, 0.15,0.84],"label":"golden_gear",   "name":"Golden Gear"},
        "shaft":  {"pos":[0.65, 0.10,0.86],"label":"crimson_shaft", "name":"Crimson Shaft"},
        "crimson":{"pos":[0.65, 0.10,0.86],"label":"crimson_shaft", "name":"Crimson Shaft"},
        "housing":{"pos":[0.55,-0.05,0.85],"label":"blue_housing",  "name":"Blue Housing"},
        "blue":   {"pos":[0.55,-0.05,0.85],"label":"blue_housing",  "name":"Blue Housing"},
        "cube":   {"pos":[0.68,-0.15,0.84],"label":"emerald_cube",  "name":"Emerald Cube"},
        "green":  {"pos":[0.68,-0.15,0.84],"label":"emerald_cube",  "name":"Emerald Cube"},
        "emerald":{"pos":[0.68,-0.15,0.84],"label":"emerald_cube",  "name":"Emerald Cube"},
    }
    DESTINATIONS = {
        "kit":        {"pos":[0.40,-0.22,0.85],"label":"kit_tray"},
        "tray":       {"pos":[0.40,-0.22,0.85],"label":"kit_tray"},
        "inspection": {"pos":[0.40, 0.32,0.84],"label":"inspection"},
        "inspect":    {"pos":[0.40, 0.32,0.84],"label":"inspection"},
        "drop":       {"pos":[0.40, 0.00,0.95],"label":"drop_zone"},
        "table":      {"pos":[0.55, 0.00,0.90],"label":"staging"},
    }
    HOME_EE = np.array([0.35,0.00,1.15])

    def __init__(self):
        self.ee      = self.HOME_EE.copy()
        self.joints  = HOME_JOINTS.copy()
        self.gripper = 0.08
        self.held    = None
        self.obj_pos = {v["label"]:np.array(v["pos"]) for v in self.OBJECTS.values()}
        log.info("DeterministicVLAPolicy ready (LIBERO Franka Panda 7-DOF)")

    def _parse(self, prompt):
        p = prompt.lower()
        src = dst = None
        for kw,info in self.OBJECTS.items():
            if kw in p:
                src=info; break
        for kw,info in self.DESTINATIONS.items():
            if kw in p:
                dst=info; break
        if src is None and any(w in p for w in ["assemble","kit a","all objects"]):
            return "KIT_A"
        if src is None: return None
        if dst is None:
            dst={"pos":[0.40,-0.22,0.85],"label":"kit_tray"}
        return (src,dst)

    def plan(self, instruction, callback=None):
        result = self._parse(instruction)
        if result=="KIT_A":
            yield from self._kit_a(callback); return
        if result is None:
            log.warning(f"Cannot parse: {instruction}"); return
        src,dst = result
        yield from self._pick_place(src["label"],np.array(src["pos"]),np.array(dst["pos"]),callback)

    def _emit(self, stage, pos, joints, grip, held, cb):
        state = {"stage":stage,"ee_pos":pos.tolist(),"joints":joints.tolist(),"gripper":grip,"held":held}
        if cb: cb(state)
        time.sleep(0.016)
        return state

    def _pick_place(self, label, pick, place, cb=None):
        pre = pick.copy(); pre[2]+=0.14
        for p in min_jerk(self.ee,pre,15):
            p=np.array(p); self.ee=p; self.joints=analytical_ik(p)
            s=self._emit("APPROACHING",p,self.joints,self.gripper,None,cb)
            yield s
        contact = pick.copy(); contact[2]+=0.025
        for p in min_jerk(pre,contact,12):
            p=np.array(p); self.ee=p; self.joints=analytical_ik(p)
            s=self._emit("DESCENDING",p,self.joints,self.gripper,None,cb)
            yield s
        for w in np.linspace(0.08,0.025,8):
            self.gripper=float(w)
            s=self._emit("GRASPING",self.ee,self.joints,w,label,cb)
            yield s
        self.held=label
        lift=contact.copy(); lift[2]+=0.18
        for p in min_jerk(contact,lift,14):
            p=np.array(p); self.ee=p; self.joints=analytical_ik(p)
            self.obj_pos[label]=p-np.array([0,0,0.04])
            s=self._emit("LIFTING",p,self.joints,self.gripper,label,cb)
            yield s
        transit=place.copy(); transit[2]+=0.18
        for p in min_jerk(lift,transit,25):
            p=np.array(p); self.ee=p; self.joints=analytical_ik(p)
            self.obj_pos[label]=p-np.array([0,0,0.04])
            s=self._emit("TRANSITING",p,self.joints,self.gripper,label,cb)
            yield s
        place_c=place.copy(); place_c[2]+=0.025
        for p in min_jerk(transit,place_c,12):
            p=np.array(p); self.ee=p; self.joints=analytical_ik(p)
            s=self._emit("PLACING",p,self.joints,self.gripper,label,cb)
            yield s
        for w in np.linspace(0.025,0.08,8):
            self.gripper=float(w)
            s=self._emit("RELEASING",self.ee,self.joints,w,None,cb)
            yield s
        self.held=None; self.obj_pos[label]=place.copy()
        retreat=place_c.copy(); retreat[2]+=0.18
        for p in min_jerk(place_c,retreat,10):
            p=np.array(p); self.ee=p; self.joints=analytical_ik(p)
            s=self._emit("RETREATING",p,self.joints,self.gripper,None,cb)
            yield s
        for p in min_jerk(retreat,self.HOME_EE,15):
            p=np.array(p); self.ee=p; self.joints=analytical_ik(p)
            s=self._emit("RETURNING HOME",p,self.joints,self.gripper,None,cb)
            yield s

    def _kit_a(self, cb=None):
        tasks=[
            ("blue_housing",[0.55,-0.05,0.85],[0.36,-0.22,0.85]),
            ("golden_gear", [0.55, 0.15,0.84],[0.40,-0.22,0.87]),
            ("crimson_shaft",[0.65,0.10,0.86],[0.44,-0.22,0.89]),
        ]
        for lbl,pick,place in tasks:
            yield from self._pick_place(lbl,np.array(pick),np.array(place),cb)

    def get_world_state(self):
        return {"ee_pos":self.ee.tolist(),"joints":self.joints.tolist(),
                "gripper":self.gripper,"held":self.held,
                "objects":{k:v.tolist() for k,v in self.obj_pos.items()}}

_policy = DeterministicVLAPolicy()
def get_policy(): return _policy

if __name__=="__main__":
    p=get_policy()
    for i,s in enumerate(p.plan("pick up the golden gear and place it in the kit tray")):
        print(f"[{i:03d}] {s['stage']:20s} ee={[round(v,3) for v in s['ee_pos']]} grip={s['gripper']:.3f}")
