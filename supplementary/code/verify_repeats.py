import json
import sys
from pathlib import Path
import numpy as np
from joint_information import ROOT, IH, information
from matrix_information import information as pair_information

def main():
    rows=[]
    for phi in [.03,.01,.001]:
        cases=[('balanced',(1.,0.,1.),np.sqrt(IH)/6,'balanced'),
               ('positive_shift',(1.,.5,1.),1.,'shift'),
               ('negative_shift',(1.,-.75,1.),1.,'shift'),
               ('unequal',(0.4,.3,2.),1.,'zero')]
        for name,H,ratio,mean in cases:
            s=ratio*phi**2.5
            alpha=1.5*s if mean=='balanced' else (2*s*H[1] if mean=='shift' else 0.)
            one=information(phi,s,alpha,H=H,gh=22,gl=36)
            two=pair_information(alpha=alpha,phi=phi,v=s*H[0],w=s*H[2],kappa=s*H[1],
                    model='Beta',gh=14,gl=12,score_method='price')
            info=two['information'][1]['efficient_kappa']
            row=dict(case=name,phi=phi,s=s,H=H,alpha=alpha,
                r1_information=one['efficient_information_gram'],r2_information=info,
                r1_per_record=one['efficient_information_gram'],r2_per_record=info/2,
                pair_mass=two['joint_mass'],pair_score_means=two['mean_score_r2'],
                pair_min_eigenvalue=min(two['information'][1]['eigenvalues']))
            if phi==.001:
                refined=pair_information(alpha=alpha,phi=phi,v=s*H[0],w=s*H[2],kappa=s*H[1],
                    model='Beta',gh=22,gl=18,score_method='price')
                row['r2_refined']=refined['information'][1]['efficient_kappa']
                row['refinement_relative_difference']=row['r2_refined']/info-1
            rows.append(row)
    (ROOT/'repeat_comparison.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(dict(rows=len(rows),max_pair_mass_error=max(abs(r['pair_mass']-1) for r in rows),
        max_pair_score_mean=max(abs(v) for r in rows for v in r['pair_score_means']),
        max_refinement_relative_difference=max(abs(r.get('refinement_relative_difference',0)) for r in rows)),indent=2))
    for row in rows:
        if row['phi']==.001:print(json.dumps(row))

if __name__=='__main__':
    main()
