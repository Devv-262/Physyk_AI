import sys, numpy as np
sys.path.insert(0, "/home/thinkspace/Desktop/Physyk_AI-master/workspace")
from physyk_perception import rack_detect as rd
rgb=np.load("perception_calib/dump_rack_rgb.npy"); depth=np.load("perception_calib/dump_rack_depth.npy")
print("frames", rgb.shape, depth.shape, "depth range", float(np.nanmin(depth)), float(np.nanmax(depth)))
eye=np.array([1.05,-0.15,1.00]); tgt=np.array([0.28,0.62,0.30])
f=tgt-eye; f/=np.linalg.norm(f); r=np.cross(f,[0,0,1.0]); r/=np.linalg.norm(r); u=np.cross(r,f)
T=np.eye(4); T[:3,:3]=np.column_stack([r,u,-f]); T[:3,3]=eye
fx=12.0/20.955*960
intr={"fx":fx,"fy":fx,"cx":480.0,"cy":270.0,"width":960,"height":540}
shelf=dict(tops=[0.08,0.36,0.64],y0s=[0.44,0.53,0.62],y1=0.86,x0=-0.05,x1=0.65)
truth={"flask_blue":(0.2568,0.5978,0.361),"flask_red":(0.0527,0.599,0.361),"beaker_red":(0.2495,0.5131,0.081),"beaker_green":(0.2499,0.6962,0.641)}
dets=rd.detect_rack_vessels(rgb[:,:,:3],depth,intr,T,shelf)
print("detections:",len(dets))
for d in dets:
    t=truth.get(d["key"])
    err=None if t is None else 1000*np.hypot(d["pos"][0]-t[0],d["pos"][1]-t[1])
    print("DET",d["key"],d["kind"],d["liquid"],d["tier_name"],"pos",np.round(d["pos"],3),"h=%.3f"%d["height_m"],"n_px",d["n_px"],"bbox",d["bbox_px"],"xy err mm:",None if err is None else round(err,1))
print("truth keys not detected:",[k for k in truth if k not in [d["key"] for d in dets]])
