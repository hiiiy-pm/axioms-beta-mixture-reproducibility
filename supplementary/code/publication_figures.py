from pathlib import Path
from PIL import Image
import json
import numpy as np
import sympy as sp
from numpy.polynomial.legendre import leggauss
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
from score_limit_audit import ExactScore, IH

ROOT = Path(__file__).resolve().parents[1] / 'results'
OUT = Path(__file__).resolve().parents[2] / 'figures'
OUT.mkdir(exist_ok=True)
COLORS = {'normal': '#55565B', 'gp': '#BB8429', 'pp': '#256A9B',
          'concentrated': '#824566', 'diffuse': '#177B7B'}
plt.rcParams.update({
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix',
    'font.size': 10, 'axes.titlesize': 10.6, 'axes.labelsize': 10,
    'xtick.labelsize': 9.4, 'ytick.labelsize': 9.4, 'legend.fontsize': 9.2,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.linewidth': .65, 'xtick.major.width': .65, 'ytick.major.width': .65,
    'lines.linewidth': 1.5, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelcolor': '#222222', 'text.color': '#222222',
    'savefig.facecolor': 'white', 'figure.facecolor': 'white',
})


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8'))


def save(fig, name):
    name = {'information_concentration': 'figure1', 'calibration_evidence': 'figure2', 'replication_geometry': 'figure3'}[name]
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / (name + '.' + ext), dpi=600, bbox_inches='tight', pad_inches=.07)
        if ext == 'png':
            with Image.open(OUT / (name + '.png')) as raster:
                rgb = raster.convert('RGB')
            rgb.save(OUT / (name + '.png'), dpi=(600, 600))
    plt.close(fig)


def panel(ax, letter, title):
    ax.set_title(rf'$\mathbf{{({letter})}}$  '+title, loc='left', pad=11)


def wilson(p, n):
    z = norm.ppf(.975)
    den = 1 + z*z/n
    mid = (p + z*z/(2*n))/den
    half = z*np.sqrt(p*(1-p)/n + z*z/(4*n*n))/den
    return float(mid-half), float(mid+half)


def window_curves(obj, edges, order):
    gx, gw = leggauss(order)
    lo = np.r_[0., edges[:-1]]
    pos = (lo[:, None]+edges[:, None])/2 + (edges-lo)[:, None]*gx[None, :]/2
    weights = (edges-lo)[:, None]*gw[None, :]/2
    vals = []
    for sign in [-1, 1]:
        x = (sign*pos).ravel()
        q, scores = obj.evaluate(x)
        nuisance = np.column_stack([scores[:, :3],
            (scores[:, 5]+obj.z0*scores[:, 4])/obj.scale])
        score = scores[:, 4]-nuisance @ obj.coef-obj.mean
        vals.append((q.reshape(pos.shape), (q*score*score).reshape(pos.shape)))
    probability = np.cumsum(np.sum(weights*(vals[0][0]+vals[1][0]), axis=1))
    information = np.cumsum(np.sum(weights*(vals[0][1]+vals[1][1]), axis=1))/obj.info
    return probability, information


def limit_density(obj, z, order=800, xorder=64):
    lam = .1
    pars = obj.parameters()
    eta, b, info = pars['rare_fraction'], pars['b'], pars['info']
    xg, xw = leggauss(xorder)
    cuts = [0, .5, 1, 2, 4, 8, 16, 32, 64]
    xx, ww = [], []
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        xx.extend((lo+(hi-lo)*(xg+1)/2).tolist())
        ww.extend(((hi-lo)*xw).tolist())
    xx, ww = np.array(xx), np.array(ww)
    F = np.logaddexp(0., -xx)*np.logaddexp(0., xx)
    tg, tw = leggauss(order)
    tmax = np.sqrt(100/(1-eta))
    tt, tw = (tg+1)*tmax/2, tw*tmax/2
    phase = tt[:, None]*b*F[None, :]/(4*np.sqrt(lam*info))
    cf = np.exp(-(1-eta)*tt**2/2 + lam/4*((np.expm1(1j*phase)-1j*phase) @ ww))
    density = []
    for lo in range(0, len(z), 200):
        density.extend((np.real(np.exp(-1j*np.outer(z[lo:lo+200], tt)) @ (tw*cf))/np.pi).tolist())
    return np.array(density)


def mechanism_values():
    edges = np.unique(np.r_[np.geomspace(.02, 4e6, 200), 10.])
    z = np.linspace(-6.5, 4.2, 481)
    rows = []
    for tau, label, color, ls in [(.01, r'$\tau=0.01$', COLORS['concentrated'], '-'),
                                  (1., r'$\tau=1$', COLORS['diffuse'], '--')]:
        D0 = 9*tau*tau/16 + IH/64
        a = 2*np.sqrt(D0*2/1.5)
        obj = ExactScore(.0001, tau=tau, a=a, gh=28, gl=64)
        prob, share = window_curves(obj, edges, 24)
        pr, sh = window_curves(obj, edges, 40)
        den = limit_density(obj, z)
        den_ref = limit_density(obj, z, order=1200, xorder=96)
        j = np.flatnonzero(edges == 10)[0]
        checks = dict(window_probability_refinement=float(max(abs(pr-prob))),
                      window_information_refinement=float(max(abs(sh-share))),
                      density_refinement=float(max(abs(den_ref-den))),
                      total_probability=float(prob[-1]), total_information_share=float(share[-1]),
                      minimum_displayed_density=float(min(den)))
        assert checks['window_probability_refinement'] < 1e-10
        assert checks['window_information_refinement'] < 2e-5
        assert checks['density_refinement'] < 1e-9
        assert abs(prob[-1]-1) < 1e-9 and abs(share[-1]-1) < 2e-5
        assert min(den) > -1e-10
        rows.append(dict(tau=tau, a=a, label=label, color=color, linestyle=ls,
            exact_information=obj.info, limiting_information=.5,
            eta=obj.parameters()['rare_fraction'], window10_probability=float(prob[j]),
            window10_information_share=float(share[j]), window_halfwidths=edges.tolist(),
            window_probabilities=prob.tolist(), window_information_shares=share.tolist(),
            density_z=z.tolist(), limiting_density=den.tolist(), lambda_value=.1, checks=checks))
    return rows


def mechanism_figure(rows=None):
    rows = mechanism_values() if rows is None else rows
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.25, 3.45), gridspec_kw={'wspace': .31})
    fig.subplots_adjust(left=.085, right=.985, bottom=.18, top=.86)
    for row in rows:
        ax.semilogx(row['window_probabilities'], row['window_information_shares'],
                    color=row['color'], ls=row['linestyle'], label=row['label'])
        marker = 'o' if row['tau'] == .01 else 's'
        ax.plot(row['window10_probability'], row['window10_information_share'], marker,
                color=row['color'], ms=5, mec='white', mew=.65, zorder=4)
        label = row['label']+rf', $\eta={100*row["eta"]:.2f}\%$ (limit)'
        bx.plot(row['density_z'], row['limiting_density'], color=row['color'], ls=row['linestyle'], label=label)
    anchor = rows[0]['window10_probability']
    ax.vlines(anchor, 0, .74, color='#777777', lw=.8, ls=':')
    ax.annotate('66.29%', (anchor, rows[0]['window10_information_share']),
                xytext=(9, 7), textcoords='offset points', color=COLORS['concentrated'], fontsize=10.5)
    ax.annotate('1.68%', (anchor, rows[1]['window10_information_share']),
                xytext=(9, 10), textcoords='offset points', color=COLORS['diffuse'], fontsize=10.5)
    ax.text(.04, .95, r'$I_{\kappa,\mathrm{eff}}\simeq0.50$ on both paths',
            transform=ax.transAxes, va='top', fontsize=9.5)
    ax.text(.04, .85, r'$h=10$ contains $0.0500\%$ of observations',
            transform=ax.transAxes, va='top', fontsize=8.8, color='#444444')
    ax.set(xlim=(8e-7, 1.35), ylim=(-.025, 1.04),
           xlabel=r'Window probability (log scale)', ylabel='Fraction of efficient information')
    ax.set_xticks([1e-6, 1e-4, 1e-2, 1])
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis='y', color='#E7E7E7', lw=.5)
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    panel(ax, 'A', 'Information concentration')
    z = np.asarray(rows[0]['density_z'])
    bx.plot(z, norm.pdf(z), color=COLORS['normal'], ls=':', lw=1.3, label='Standard normal')
    bx.set(xlim=(-6.5, 4.2), ylim=(0, .73), xlabel='Standardized efficient score', ylabel='Density')
    bx.set_xticks([-6, -4, -2, 0, 2, 4])
    panel(bx, 'B', r'Score limits at $n\Phi\to0.1$')
    bx.legend(frameon=False, loc='upper left', fontsize=9, handlelength=2.2, labelspacing=.45)
    bx.set_yticks([0, .2, .4, .6])
    save(fig, 'information_concentration')
    return rows


def calibration_figure():
    actual = read('calibration_validation.json')
    cfrows = read('finite_bulk.json')['grid']
    original = read('joint_score_test.json')['simulation']
    benchmark = read('point_null_benchmark.json')['settings']
    shapes = [[1, 0, 1], [1, .4, 1.2], [.7, -.25, 1.4]]
    rows, matrices = [], []
    for H in shapes:
        selected = sorted([r for r in actual['scalar_grid'] if np.allclose(r['H'], H)],
                          key=lambda r: (r['tau'], r['direction']))
        rows.extend(selected)
        matrix = np.array([[100*(r['methods'][method][tail]-.025) for r in selected]
                          for method in ['normal', 'gaussian_poisson', 'finite_bulk']
                          for tail in ['left', 'right']])
        intervals = np.array([[wilson(r['methods'][method][tail], r['reps']) for r in selected]
                             for method in ['normal', 'gaussian_poisson', 'finite_bulk']
                             for tail in ['left', 'right']])
        excludes = (intervals[:, :, 1] < .025) | (intervals[:, :, 0] > .025)
        matrices.append(dict(H=H, columns=[dict(tau=r['tau'], direction=r['direction']) for r in selected],
                             signed_tail_error_pp=matrix.tolist(), wilson95=intervals.tolist(),
                             wilson_excludes_nominal=excludes.tolist()))
    fig = plt.figure(figsize=(7.25, 7.35))
    top = fig.add_gridspec(1, 2, left=.18, right=.985, bottom=.43, top=.90, wspace=.11)
    bottom = fig.add_gridspec(1, 2, left=.11, right=.985, bottom=.075, top=.29, wspace=.36)
    methods = [('normal', 'Normal', COLORS['normal'], 's', -.24),
               ('gaussian_poisson', 'GP', COLORS['gp'], '^', 0),
               ('finite_bulk', 'PP', COLORS['pp'], 'o', .24)]
    noise = float(100*norm.ppf(.975)*np.sqrt(.025*.975/16000))
    points = []
    labels = []
    for row in rows:
        hidx = next(i for i, H in enumerate(shapes) if np.allclose(row['H'], H)) + 1
        sign = {-1: '-', 0: '0', 1: '+'}[row['direction']]
        labels.append(rf'$H_{hidx},\ {row["tau"]:.2f},\ {sign}$')
    for idx, tail in enumerate(['left', 'right']):
        axis = fig.add_subplot(top[0, idx])
        axis.axvspan(-noise, noise, color='#EAECEF', zorder=0)
        axis.axvline(0, color='#555555', lw=.8, ls=':', zorder=1)
        for key, label, color, marker, offset in methods:
            estimates, lows, highs = [], [], []
            for row in rows:
                rate = row['methods'][key][tail]
                lo, hi = wilson(rate, row['reps'])
                estimates.append(100*(rate-.025)); lows.append(100*(rate-lo)); highs.append(100*(hi-rate))
                points.append(dict(H=row['H'], tau=row['tau'], direction=row['direction'],
                                   method=key, tail=tail, probability=rate, datasets=row['reps'], wilson95=[lo, hi]))
            axis.errorbar(estimates, np.arange(18)+offset, xerr=[lows, highs], ls='none',
                          color=color, marker=marker, ms=3.8, mec='white', mew=.35,
                          capsize=1.4, elinewidth=.75, label=label, zorder=3)
        for boundary in [5.5, 11.5]: axis.axhline(boundary, color='#BFC4C9', lw=.65)
        for boundary in [2.5, 8.5, 14.5]: axis.axhline(boundary, color='#E5E7E9', lw=.5)
        axis.set(ylim=(17.65, -.65), xlim=(-2.15, 2.75), xticks=[-2, -1, 0, 1, 2],
                 yticks=np.arange(18), yticklabels=labels if idx == 0 else [],
                 xlabel='Tail error (percentage points)')
        axis.tick_params(axis='y', length=0, labelsize=9.1, pad=6)
        axis.grid(axis='x', color='#E5E7E9', lw=.45, zorder=0)
        panel(axis, chr(65+idx), 'Left tail' if idx == 0 else 'Right tail')
        if idx == 0:
            axis.text(-.43, 1.035, r'Shape, $\tau$, sign$(m)$', transform=axis.transAxes, fontsize=9.2)
    handles = [Line2D([], [], color=color, marker=marker, ls='none', ms=4, label=label)
               for _, label, color, marker, _ in methods]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.57, .985), ncol=3,
               frameon=False, fontsize=10, handletextpad=.4, columnspacing=2.6)
    fig.text(.18, .346, 'Nominal tail probability 2.5%; shaded band shows pointwise Monte Carlo variation', fontsize=9)
    axis = fig.add_subplot(bottom[0, 0])
    displayed = [r for r in cfrows if r['direction'] != 0]
    assert len(displayed) == 24
    for key, label, color, ls in [('normal_error', 'Normal', COLORS['normal'], ':'),
            ('gaussian_poisson_error', 'GP', COLORS['gp'], '--'),
            ('finite_bulk_error', 'PP', COLORS['pp'], '-')]:
        values = np.sort([r[key] for r in displayed])
        axis.step(np.r_[1e-4, values, .3], np.r_[0., np.arange(1, 25)/24, 1.],
                  where='post', label=label, color=color, ls=ls)
    axis.set_xscale('log')
    axis.set(xlim=(1.8e-4, .3), ylim=(0, 1.04), xticks=[1e-3, 1e-2, 1e-1],
             xlabel='Maximum CF error (log scale)', ylabel='Fraction of 24 comparisons')
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    axis.grid(axis='y', color='#E7E7E7', lw=.5)
    axis.legend(frameon=False, fontsize=8.8, loc='lower right')
    panel(axis, 'C', r'Distributional error, $\eta>0$')
    axis = fig.add_subplot(bottom[0, 1])
    joint_points = []
    for shift, method, label, color, marker in [(-.21, 'normal', r'$\chi^2_5$', COLORS['normal'], 's'),
            (-.07, 'gp', 'GP', COLORS['gp'], '^'), (.07, 'pp', 'PP', COLORS['pp'], 'o'),
            (.21, 'null_mc', 'Null MC', '#276D53', 'D')]:
        rates, lows, highs = [], [], []
        for row, old, new in zip(actual['joint_primary'], original, benchmark):
            assert row['n'] == old['n'] == new['n']
            rate = (row['usual_size'] if method == 'normal' else old['derived_size'] if method == 'gp'
                    else row['derived_size'] if method == 'pp' else new['references']['null_mc']['size'])
            lo, hi = wilson(rate, row['reps'])
            rates.append(100*rate); lows.append(100*(rate-lo)); highs.append(100*(hi-rate))
            joint_points.append(dict(n=row['n'], method=method, rejection_probability=rate,
                                     datasets=row['reps'], wilson95=[lo, hi]))
        axis.errorbar(np.arange(3)+shift, rates, yerr=[lows, highs], color=color, marker=marker,
                      ls='none', ms=4, capsize=2, elinewidth=.8, label=label)
    axis.axhline(5, color='#555555', lw=.8, ls=':')
    axis.set(xticks=range(3), xticklabels=['200', '1,000', '10,000'], xlabel='Independent clusters',
             ylabel='Null rejection (%)', ylim=(4.05, 8.7), xlim=(-.42, 2.42))
    axis.grid(axis='y', color='#E7E7E7', lw=.5)
    axis.legend(frameon=False, fontsize=8.6, loc='upper right', ncol=2, columnspacing=.8, handletextpad=.4)
    panel(axis, 'D', 'Full-parameter point test')
    save(fig, 'calibration_evidence')
    return dict(tail_matrices=matrices, tail_points=points, nominal_pointwise_noise_band_pp=[-noise, noise],
                joint_points=joint_points, cf_displayed_count=24, cf_zero_eta_count=12,
                cf_rows=[{k:r[k] for k in ['phi','tau','H','n','direction','normal_error','gaussian_poisson_error','finite_bulk_error']} for r in cfrows])


def replication_figure():
    u, s = sp.symbols('u s')
    A=s*u/2;B=1-u;V=(u*u-2)/8;W=(u*u-3*u+1)/2;K=A*(2-u)
    def moment(expr):
        p = sp.Poly(sp.expand((expr.subs(s,1)+expr.subs(s,-1))/2), u)
        return sp.simplify(sum(c*sp.factorial(k[0]) for k,c in p.terms()))
    scores=[A,B,V,W,K]
    I1=sp.Matrix([[moment(x*y) for y in scores] for x in scores])
    u1,u2,s1,s2=sp.symbols('u1 u2 s1 s2')
    def paired_moment(expr):
        terms=sp.Poly(sp.expand(expr),u1,u2,s1,s2).terms()
        return sp.simplify(sum(c*sp.factorial(i)*sp.factorial(j)
            for (i,j,k,l),c in terms if k%2==0 and l%2==0))
    A2=(s1*u1+s2*u2)/2; B2=2-u1-u2
    paired_scores=[A2,B2,(A2*A2-1)/2,(B2*B2+B2-2)/2,A2*(B2+1)]
    I2=sp.Matrix([[paired_moment(x*y) for y in paired_scores] for x in paired_scores])
    assert I1.rank()==4 and I1.det()==0 and I2.rank()==5 and I2.det()==7
    moments = [paired_moment((A2*(B2+2))**k) for k in range(1, 5)]
    assert moments == [0, 4, 0, 1800]
    eigen1=np.linalg.eigvalsh(np.array(I1).astype(float));eigen1[0]=0.
    eigen2=np.linalg.eigvalsh(np.array(I2).astype(float))
    r1=[x for x in read('deep_followup.json')['simulation'] if x['tau']==.01]
    r2=read('pair_score_audit.json')['simulations']
    assert [x['n'] for x in r1]==[x['n_records'] for x in r2]
    fig=plt.figure(figsize=(7.25,3.25))
    grid=fig.add_gridspec(1,3,left=.07,right=.985,bottom=.18,top=.76,
                         width_ratios=[1.12,1,1],wspace=.38)
    ax=fig.add_subplot(grid[0,0])
    bx=fig.add_subplot(grid[0,1])
    cx=fig.add_subplot(grid[0,2],sharey=bx)
    for vals,shift,color,marker,label in [(eigen1,-.13,COLORS['concentrated'],'s',r'$r=1$'),
                                        (eigen2,.13,COLORS['diffuse'],'o',r'$r=2$')]:
        ax.scatter(vals,np.arange(1,6)+shift,color=color,marker=marker,s=27,label=label,zorder=3)
    ax.axvline(0,color='#999999',lw=.7)
    ax.annotate('0',xy=(0,.87),xytext=(7,-10),textcoords='offset points',fontsize=9,color=COLORS['concentrated'])
    ax.annotate(f'{eigen2[0]:.3f}',xy=(eigen2[0],1.13),xytext=(7,6),textcoords='offset points',fontsize=9,color=COLORS['diffuse'])
    ax.set(yticks=range(1,6),yticklabels=[r'$\lambda_1$',r'$\lambda_2$',r'$\lambda_3$',r'$\lambda_4$',r'$\lambda_5$'],
           xlabel='Score eigenvalue',ylim=(.5,5.6),xlim=(-.15,6.0),xticks=[0,2,4,6])
    ax.grid(axis='y',color='#E7E7E7',lw=.5)
    panel(ax,'A','Limiting eigenvalues')
    points=[]
    for rr,rows,color,marker in [(1,r1,COLORS['concentrated'],'s'),(2,r2,COLORS['diffuse'],'o')]:
        for tail,axis in [('left',bx),('right',cx)]:
            vals=[];low=[];high=[]
            for row in rows:
                rate=row[tail+'_tail'];lo,hi=wilson(rate,row['reps'])
                vals.append(100*rate);low.append(100*(rate-lo));high.append(100*(hi-rate))
                records=row['n'] if rr==1 else row['n_records']
                points.append(dict(r=rr,records=records,tail=tail,probability=rate,datasets=row['reps'],wilson95=[lo,hi]))
            axis.errorbar(np.arange(3)+(-.10 if rr==1 else .10),vals,yerr=[low,high],color=color,marker=marker,
                mfc=color,mec='white',mew=.45,ls='none',ms=5,capsize=2.5,
                elinewidth=.9,label=rf'$r={rr}$',zorder=3)
    for axis in [bx,cx]:
        axis.axhline(2.5,color='#555555',lw=.85,ls=':')
        axis.set(xticks=range(3),xticklabels=['200','1,000','10,000'],xlabel='Total records, $N$',
                 ylim=(0,6.5),xlim=(-.4,2.4),yticks=[0,2.5,5])
        axis.grid(axis='y',color='#E7E7E7',lw=.5)
    bx.set_ylabel('Tail probability (%)')
    cx.tick_params(labelleft=False)
    panel(bx,'B','Left tail')
    panel(cx,'C','Right tail')
    handles=[Line2D([],[],color=COLORS['concentrated'],marker='s',ls='none',ms=5,
                   label=r'$r=1$ (rank 4)'),
             Line2D([],[],color=COLORS['diffuse'],marker='o',ls='none',ms=5,
                   label=r'$r=2$ (rank 5)')]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.51,.99),ncol=2,
               frameon=False,fontsize=10,handletextpad=.5,columnspacing=2.6)
    save(fig,'replication_geometry')
    return dict(I1=[[str(x) for x in row] for row in I1.tolist()],
                I2=[[str(x) for x in row] for row in I2.tolist()],
                eigenvalues_r1=eigen1.tolist(),eigenvalues_r2=eigen2.tolist(),tail_points=points,
                limiting_efficient_moments=[int(x) for x in moments], limiting_skewness=0,
                limiting_kurtosis=112.5, standardized_sum_excess_kurtosis_coefficient=109.5,
                limiting_information_per_record=2)


def main():
    out={'mechanism':mechanism_figure(),'calibration':calibration_figure(),'replication':replication_figure(),
         'intervals':'Wilson 95% binomial Monte Carlo intervals, conditional on independently computed reference quantiles.',
         'new_calculations':'Exact-window beta integration, Gaussian-Poisson Fourier inversion, and exact replicated-score moments; calibration panels read complete archived simulations and the independent known-null benchmark.'}
    (ROOT/'publication_figures.json').write_text(json.dumps(out,indent=2),encoding='utf-8')
    print(json.dumps({'mechanism':[{k:r[k] for k in ['tau','exact_information','eta','window10_probability','window10_information_share','checks']} for r in out['mechanism']],
                      'smallest_eigenvalues':[out['replication']['eigenvalues_r1'][0],out['replication']['eigenvalues_r2'][0]],
                      'figures':['information_concentration','calibration_evidence','replication_geometry']},indent=2))


if __name__=='__main__':
    main()
