"""Per-pick joint statistics from e2e_batch.py results + workspace/joint_trace.csv: peak joint speed vs Franka's caps, frames at the cap,
acceleration vs the rated limits, joint path, and frames where the hand is nearly still while a joint moves fast (swivel).
  usage: .venv/bin/python workspace/scripts/arm_tests/analyze_e2e.py <results.json> [more.json ...]
The trace file must be the one written while the batch ran (the sim truncates it at start)."""
import json, csv, sys, numpy as np
from pathlib import Path
TRACE=str(Path(__file__).resolve().parents[3] / "workspace" / "joint_trace.csv")
CAP=np.array([2.175]*4+[2.61]*3); ACC=np.array([15,7.5,10,12.5,15,20,20.]); DT=1/60.
rows=list(csv.reader(open(TRACE))); H=rows[0]; R=rows[1:]; col={h:i for i,h in enumerate(H)}
def A(n,a,b): return np.array([float(r[col[n]]) if r[col[n]]!="nan" else np.nan for r in R[a:b]])
def analyze(files):
    out=[]
    for fn in files:
        for r in json.load(open(fn)):
            a,b=r["rows"][0]-1, r["rows"][1]-1       # data-row indices
            if b-a<50: continue
            QD=np.array([A(f"qd{i}",a,b) for i in range(1,8)]); Q=np.array([A(f"q{i}",a,b) for i in range(1,8)]); ev=A("event",a,b).astype(int)
            hand=np.array([A("hand_x",a,b),A("hand_y",a,b),A("hand_z",a,b)]).T
            hs=np.r_[0,np.linalg.norm(np.diff(hand,axis=0),axis=1)/DT]
            pk=np.nanmax(np.abs(QD)/CAP[:,None],axis=1); atcap=int(np.sum(np.max(np.abs(QD[:6])/CAP[:6,None],axis=0)>0.97))
            acc=np.abs(np.diff(QD,axis=1)/DT); pa=np.nanmax(acc/ACC[:,None],axis=1); over=float(np.mean(np.any(acc[:6]>ACC[:6,None],axis=0))*100)
            path=np.sum(np.abs(np.diff(Q,axis=1)),axis=1); sw=int(np.sum((hs<0.05)&(np.nanmax(np.abs(QD[:6]),axis=0)>1.0)))
            worst=int(np.nanargmax(np.max(np.abs(QD[:6])/CAP[:6,None],axis=0)))
            out.append(dict(mode=r["mode"], layout=r["layout"], surface=r["surface"], ins=r["instruction"].replace("pick up the ","").replace(" and place it",""), ok=r["ok"], err=r["err"], wall=r["wall"], rows=b-a,
                            pk=float(pk[:6].max()), pk7=float(pk[6]), atcap=atcap, pa=float(pa[:6].max()), over=over, path=float(path[:6].sum()), sw=sw, worst_phase=int(ev[worst])))
    return out
if __name__=="__main__":
    res=analyze(sys.argv[1:])
    print(f"{'mode':7s} L surf  {'pick':44s} ok   err_mm wall | max|qd|/cap j1-6 | atCap | accel/lim | %fr>lim | j1-6 path | swivel fr | phase of peak")
    for r in res:
        print(f"{r['mode']:7s} {r['layout']} {r['surface']:5s} {r['ins'][:44]:44s} {'Y' if r['ok'] else 'N'} {r['err'] if r['err'] is not None else float('nan'):7.1f} {r['wall']:5.1f} | {r['pk']:5.2f} | {r['atcap']:3d} | {r['pa']:5.2f} | {r['over']:4.1f} | {r['path']:5.1f} | {r['sw']:3d} | {r['worst_phase']}")
    import collections
    for m in sorted(set(r['mode'] for r in res)):
        g=[r for r in res if r['mode']==m]
        print(f"\n[{m}] picks {len(g)}  success {sum(1 for r in g if r['ok'])}/{len(g)}  mean wall {np.mean([r['wall'] for r in g]):.1f}s  mean err {np.mean([r['err'] for r in g if r['err'] is not None]):.1f}mm | peak |qd|/cap: mean {np.mean([r['pk'] for r in g]):.2f} max {max(r['pk'] for r in g):.2f} | frames at cap total {sum(r['atcap'] for r in g)} | swivel frames {sum(r['sw'] for r in g)} | mean j1-6 path {np.mean([r['path'] for r in g]):.1f} rad")
