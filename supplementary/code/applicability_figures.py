import sys
sys.dont_write_bytecode = True
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.lines import Line2D
from publication_figures import COLORS, panel


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / 'figures'


def interval(ax, x, row, color, marker, label=None):
    p = 100 * row['size']
    lo, hi = np.array(row['wilson95']) * 100
    ax.errorbar(x, p, yerr=[[p-lo],[hi-p]], color=color, marker=marker, markersize=4.2,
                capsize=2, elinewidth=.8, lw=0, label=label, markerfacecolor='white' if marker=='s' else color)


def save(fig, name):
    for ext in ['pdf', 'png']:
        fig.savefig(OUT / (name + '.' + ext), dpi=600, bbox_inches='tight', pad_inches=.08)
    with Image.open(OUT / (name + '.png')) as im:
        rgb = im.convert('RGB')
    rgb.save(OUT / (name + '.png'), dpi=(600,600))
    plt.close(fig)


def main():
    source = ROOT / 'results' / 'applicability_experiments.json'
    r = json.loads(source.read_text())
    regionfile = ROOT / 'results' / 'joint_region.json'
    region = json.loads(regionfile.read_text())
    methods = [('Chi-square', COLORS['normal'], 'x', r'$\chi^2_5$'),
               ('GP', COLORS['gp'], '^', 'GP'), ('PP', COLORS['pp'], 'o', 'PP'),
               ('MC', COLORS['diffuse'], 's', 'Null MC')]
    fig, axes = plt.subplots(2,2,figsize=(7.2,5.8),sharex=True,sharey=True)
    phis = r['design']['precision_grid']
    for index, (ax, (n, tau)) in enumerate(zip(axes.ravel(), [(200,.01),(200,.3),(1000,.01),(1000,.3)])):
        panel(ax, 'ABCD'[index], rf'$n={n},\ \tau={tau}$')
        for j, phi in enumerate(phis):
            row = next(t for t in r['precision'] if t['phi']==phi and t['n']==n and t['tau']==tau)
            for k, (key,color,marker,label) in enumerate(methods):
                interval(ax, j+(k-1.5)*.13, row['references'][key], color, marker)
        ax.axhline(5,color='#84858A',ls=':',lw=1,zorder=0)
        ax.set_ylim(3.7,8.9)
        ax.set_yticks([4,5,6,7,8])
        ax.set_xticks(range(5),[r'$10^{-4}$',r'$10^{-3}$',r'$10^{-2}$','0.05','0.1'])
        ax.grid(axis='y',alpha=.12)
        if index in [0,2]: ax.set_ylabel('Rejection probability (%)')
        if index in [2,3]: ax.set_xlabel(r'Baseline precision $\Phi$')
    handles=[Line2D([0],[0],color=c,marker=m,lw=0,label=l,markersize=5,
                    markerfacecolor='white' if m=='s' else c) for _,c,m,l in methods]
    fig.legend(handles=handles,loc='upper center',bbox_to_anchor=(.5,1.005),ncol=4,frameon=False)
    fig.subplots_adjust(left=.10,right=.99,bottom=.09,top=.89,hspace=.36,wspace=.17)
    save(fig,'figure4')
    fig, axes = plt.subplots(2,2,figsize=(7.2,6.0))
    ax=axes[0,0]
    panel(ax,'A','Residual dependence in pairs')
    for j, p in enumerate(r['pair']):
        color=[COLORS['pp'],COLORS['concentrated']][j]
        vals=[]
        for i,row in enumerate(p['rows'][:3]):
            x=i*.25+(j-.5)*.018
            interval(ax,x,row['normal'],color,['o','s'][j])
            vals.append(row['normal']['size']*100)
        ax.plot([0,.25,.5],vals,color=color,lw=.8,alpha=.6)
        ax.plot([],[],color=color,marker=['o','s'][j],label=rf'$\Phi={p["phi"]}$',markersize=4)
    ax.axhline(5,color='#84858A',ls=':',lw=1)
    ax.set_xticks([0,.25,.5]);ax.set_ylim(0,27)
    ax.set_xlabel('Latent-normal copula correlation')
    ax.set_ylabel('Rejection probability (%)')
    ax.legend(loc='upper left',frameon=False,fontsize=8.8)
    ax=axes[0,1]
    panel(ax,'B','Finite recording resolution')
    for digits,color,ls in [(2,COLORS['normal'],':'),(4,COLORS['pp'],'-'),(6,COLORS['diffuse'],'--')]:
        values=[x for x in r['resolution'] if x['digits']==digits]
        ax.loglog([x['phi'] for x in values],[x['expected_interior_records'] for x in values],
                  color=color,ls=ls,marker='o',ms=3,label=f'{digits} decimal places')
    ax.set_xlabel(r'Baseline precision $\Phi$')
    ax.set_ylabel('Expected non-endpoint records')
    ax.text(.04,.94,r'$N=1000$',transform=ax.transAxes,va='top',fontsize=9)
    ax.legend(loc='lower right',frameon=False,fontsize=8.2,handlelength=1.7)
    ax=axes[1,0]
    panel(ax,'C','Local covariance alternatives')
    for i,row in enumerate(r['power']):
        null=next(t for t in r['precision'] if t['phi']==row['phi'] and t['tau']==.3 and t['n']==1000)
        interval(ax,i-.08,null['references']['PP'],COLORS['normal'],'s')
        interval(ax,i+.08,row['references']['PP'],COLORS['pp'],'o')
    ax.axhline(5,color='#84858A',ls=':',lw=1)
    ax.set_xticks([0,1,2],[r'$10^{-4}$',r'$10^{-2}$','0.1'])
    ax.set_ylim(3.8,6.5)
    ax.set_ylabel('PP rejection probability (%)')
    ax.set_xlabel(r'Baseline precision $\Phi$')
    handles=[Line2D([0],[0],color=COLORS['normal'],marker='s',lw=0,mfc='white',label=r'$\rho=0$'),
             Line2D([0],[0],color=COLORS['pp'],marker='o',lw=0,label=r'$\rho=0.5$')]
    ax.legend(handles=handles,loc='upper right',ncol=2,frameon=False,fontsize=8.6)
    ax=axes[1,1]
    panel(ax,'D','Joint-region inversion')
    candidates={t['index']:t for t in region['candidates']}
    for grid,color,marker,ls in [(region['grids'][1],COLORS['pp'],'o','-'),
                                 (region['grids'][0],COLORS['normal'],'s','')]:
        x,y=[],[]
        rows=[candidates[i] for i in grid['indices']]
        for phi in grid['phi_grid']:
            selected=[t for t in rows if abs(t['factors'][0]-phi)<1e-10]
            x.append(phi); y.append(min(-t['margin'] for t in selected))
        ax.plot(x,y,color=color,marker=marker,ls=ls,ms=4,mfc='white' if marker=='s' else color,
                label=f'{grid["count"]} nodes')
    ax.axhline(0,color='#84858A',ls=':',lw=1)
    ax.set_xlabel(r'Candidate precision $\Phi/\Phi_0$')
    ax.set_ylabel(r'Grid minimum of $T_n-c_{0.95}$')
    ax.set_xticks([.8,.9,1,1.1,1.2])
    ax.legend(loc='upper right',frameon=False,fontsize=8.5)
    ax.text(.50,.62,'Covariance hull\n'+r'$[-0.36,0.36]\Phi_0^{5/2}$',
            transform=ax.transAxes,ha='center',fontsize=9,linespacing=1.4)
    for ax in axes.ravel(): ax.grid(axis='y',alpha=.12)
    fig.subplots_adjust(left=.10,right=.99,bottom=.09,top=.94,hspace=.48,wspace=.38)
    save(fig,'figure5')


if __name__ == '__main__':
    main()
