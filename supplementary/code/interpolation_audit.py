from pathlib import Path
import json
import numpy as np
from scipy.interpolate import PchipInterpolator
from score_limit_audit import ExactScore,IH
from joint_information import data_quadrature

ROOT=Path(__file__).resolve().parents[1] / 'results'


def audit(obj):
    mass=obj.mass/sum(obj.mass)
    basis=np.column_stack([obj.S[:,[0,1,2,4]],obj.S[:,5]/obj.scale])
    means=mass@basis
    gram=basis.T@(mass[:,None]*basis)
    invL=np.linalg.inv(np.linalg.cholesky(gram)).T
    interp=PchipInterpolator(obj.x,(basis-means)@invL,axis=0,extrapolate=False)
    x,dx=data_quadrature(obj.phi,gl=37,central_cut=64,tail_cut=120)
    density,S=obj.evaluate(x)
    B=np.column_stack([S[:,[0,1,2,4]],S[:,5]/obj.scale])
    d=interp(x)-(B-means)@invL
    scalar=(obj.interp(x)-obj.efficient(x))/np.sqrt(obj.info)
    valid=np.all(np.isfinite(d),axis=1)&np.isfinite(scalar)
    d[~valid]=0;scalar[~valid]=0
    weights=dx*density
    central=np.linspace(-30,30,1201)
    _,CS=obj.evaluate(central)
    CB=np.column_stack([CS[:,[0,1,2,4]],CS[:,5]/obj.scale])
    cd=interp(central)-(CB-means)@invL
    cs=(obj.interp(central)-obj.efficient(central))/np.sqrt(obj.info)
    return dict(phi=obj.phi,tau=obj.tau,a=obj.a,H=obj.H,integration_order=37,
        quadrature_mass=float(weights.sum()),
        scalar_L2_error=float(np.sqrt(weights@(scalar*scalar))),
        scalar_mean_error=float(weights@scalar),
        joint_coordinate_L2_errors=np.sqrt(weights@(d*d)).tolist(),
        joint_coordinate_mean_errors=(weights@d).tolist(),
        central_scalar_max_absolute_error=float(abs(cs).max()),
        central_joint_max_absolute_error=float(abs(cd).max()))


def main():
    grid=json.loads((ROOT/'finite_bulk.json').read_text())['grid']
    rows=[]
    tau=.01;d0=IH/64+9*tau*tau/16;a=2*np.sqrt(d0*2/1.5)
    rows.append(audit(ExactScore(.0001,tau=tau,a=a,gh=20,gl=48)))
    for r in grid:
        if r['n']==1000:
            rows.append(audit(ExactScore(r['phi'],tau=r['tau'],a=r['a'],H=r['H'],gh=20,gl=48)))
    out=dict(rows=rows,summary=dict(settings=len(rows),
        max_scalar_L2_error=max(r['scalar_L2_error'] for r in rows),
        max_joint_coordinate_L2_error=max(max(r['joint_coordinate_L2_errors']) for r in rows),
        max_absolute_joint_coordinate_mean_error=max(max(abs(x) for x in r['joint_coordinate_mean_errors']) for r in rows),
        max_central_scalar_error=max(r['central_scalar_max_absolute_error'] for r in rows),
        max_central_joint_error=max(r['central_joint_max_absolute_error'] for r in rows)))
    (ROOT/'interpolation_audit.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps(out['summary'],indent=2))


if __name__=='__main__':main()
