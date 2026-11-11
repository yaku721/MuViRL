from pathlib import Path
import argparse
import csv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT/'representation'/name).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def main(output):
    output.mkdir(parents=True,exist_ok=True)
    families={font.name for font in font_manager.fontManager.ttflist}
    font='Times New Roman' if 'Times New Roman' in families else 'DejaVu Serif'
    plt.rcParams.update({'font.family':font, 'font.size':10, 'axes.labelsize':10,
                         'axes.titlesize':11, 'pdf.fonttype':42, 'ps.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(10,8),layout='constrained')
    for ax,name,title,color,limits in [
            (axes[0,0],'unit_consistency.csv','(a) Unit consistency','#2878B5',(.3,1.)),
            (axes[0,1],'view_consistency.csv','(b) View consistency','#D64A54',(.55,1.))]:
        rows=read(name)
        ax.plot(limits,limits,'--',color='#6D7781',lw=.9,zorder=1)
        ax.scatter([float(r['muvirl_similarity']) for r in rows],
                   [float(r['baseline_similarity']) for r in rows],
                   s=17,color=color,alpha=.85,edgecolors='white',linewidths=.25,zorder=2)
        ax.set(xlim=limits,ylim=limits,xlabel='MuViRL similarity',ylabel='Baseline similarity',
               title=title,aspect='equal')
        ax.grid(alpha=.15)
        ax.set_axisbelow(True)
    rows=[r for r in read('temporal_similarity.csv') if r['pair_type']=='audiovisual']
    length=1+max(int(r['query_frame']) for r in rows)
    matrix=[[0.]*length for _ in range(length)]
    for r in rows: matrix[int(r['query_frame'])][int(r['candidate_frame'])]=float(r['cosine_similarity'])
    ax=axes[1,0]
    im=ax.imshow(matrix,origin='lower',cmap='viridis',vmin=-.1,vmax=1.,interpolation='nearest')
    ax.plot([0,length-1],[0,length-1],'--',color='#343A40',lw=.65)
    ax.set(xlabel='Synthesized visual frame',ylabel='Audio frame',title='(c) Cross-modal correspondence')
    fig.colorbar(im,ax=ax,fraction=.046,pad=.04,label='Cosine similarity')
    profiles=read('temporal_offsets.csv')
    ax=axes[1,1]
    ax.axvline(0,color='#7A838C',linestyle=':',lw=.8)
    for pair,label,color,style,marker in [
            ('visual','Visual','#2878B5','-','o'),
            ('audiovisual','Audiovisual','#D64A54','--','s')]:
        rows=sorted([r for r in profiles if r['pair_type']==pair],key=lambda r:int(r['frame_offset']))
        ax.plot([int(r['frame_offset']) for r in rows],[float(r['mean_similarity']) for r in rows],
                label=label,color=color,ls=style,lw=1.5,marker=marker,markevery=5,markersize=4,
                markerfacecolor='white',markeredgewidth=.8)
    ax.set(xlim=(-25,25),ylim=(0,1),xlabel='Frame offset',ylabel='Mean similarity',title='(d) Temporal alignment')
    ax.legend(loc='upper right',frameon=True,edgecolor='#707070',fancybox=False)
    ax.grid(alpha=.15)
    ax.set_axisbelow(True)
    fig.savefig(output/'representation_structure.pdf')
    fig.savefig(output/'representation_structure.png',dpi=300)
    plt.close(fig)
    print(f'Plots saved to {output.resolve()}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('plots'))
    main(parser.parse_args().output)
