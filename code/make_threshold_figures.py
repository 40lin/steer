"""Plot measured simulation outputs only; no synthetic demonstration points."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'experiments'/'dynamic_threshold'
ANALYSIS=OUT/'analysis'/'test'
FIG=OUT/'figures'

def require_two_panel_alignment(fig, axes, json_out, tolerance_pt=1.5):
    """Check this figure's two side-by-side plot rectangles after rendering.

    This small local check has no external skill/runtime dependency. It checks
    shared top/bottom edges, equal plot widths/heights, and matching panel-label
    offsets in physical points. It is not a general-purpose collision audit.
    """
    if len(axes) != 2:
        raise ValueError('This alignment check requires exactly two panels')
    fig.canvas.draw()
    scale=72.0/fig.dpi
    rectangles=[]
    label_offsets=[]
    for axis,label in zip(axes,('a','b')):
        box=axis.get_window_extent()
        rectangles.append([float(x*scale) for x in (box.x0,box.y0,box.x1,box.y1)])
        labels=[text for text in axis.texts if text.get_text()==label]
        if len(labels)!=1:
            raise ValueError(f'Expected exactly one panel label {label!r}')
        point=labels[0].get_transform().transform(labels[0].get_position())
        label_offsets.append([float((point[0]-box.x0)*scale),
                              float((point[1]-box.y1)*scale)])
    left,right=np.asarray(rectangles)
    differences={
        'bottom_edge':abs(left[1]-right[1]),
        'top_edge':abs(left[3]-right[3]),
        'plot_width':abs((left[2]-left[0])-(right[2]-right[0])),
        'plot_height':abs((left[3]-left[1])-(right[3]-right[1])),
        'panel_label_horizontal_offset':abs(label_offsets[0][0]-label_offsets[1][0]),
        'panel_label_vertical_offset':abs(label_offsets[0][1]-label_offsets[1][1]),
    }
    report={'status':'pass' if max(differences.values())<=tolerance_pt else 'fail',
            'scope':'Two side-by-side panels; geometry and label anchors only',
            'tolerance_pt':tolerance_pt,'plot_rectangles_pt':rectangles,
            'label_offsets_pt':label_offsets,
            'differences_pt':{key:float(value) for key,value in differences.items()}}
    Path(json_out).write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report['status']!='pass':
        raise ValueError(f'Panel geometry is misaligned; see {json_out}')
    return report

def records(path):
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

if __name__=='__main__':
    summaries=json.loads((ANALYSIS/'results_summary.json').read_text(encoding='utf-8'))
    rows={row['method']:row for row in records(ANALYSIS/'paper_table.csv')}
    seeds=records(ANALYSIS/'seed_results.csv')
    methods=['belief','history','openloop','schedule_best','bayes_greedy','fete_fixed','ucb']
    names=['Posterior + GRU','History-only GRU','Time-only GRU','One-switch schedule',
           'Posterior greedy','Probe, then fixed','Revenue UCB']
    primary=summaries['primary']
    FIG.mkdir(exist_ok=True)
    contract={
       'question':'Can a dynamic threshold increase receipts while retaining 95% of no-threshold rented core-hours?',
       'claim':('The primary controller meets the predefined joint revenue/service criterion in this finite learner catalogue.'
                if primary['joint_confirmatory_success'] else
                'The primary controller does not meet the predefined joint revenue/service criterion in this finite learner catalogue.'),
       'archetype':'Paired effect with a necessary service-constraint boundary',
       'panel_a':'Economic benefit relative to validation-selected fixed rho=0.5',
       'panel_b':'Potential falsifier: rented core-hours minus 0.95 times no-threshold baseline',
       'data':'Synthetic market experiments, with actual recorded outcomes; not empirical cloud measurements',
       'uncertainty':'Two-sided percentile 95% crossed-bootstrap intervals, 10000 resamples; one-sided 95% lower bounds determine the prespecified formal decision',
       'replicates':'10 independent training seeds for each GRU; 100 fresh worlds per each of 16 fixed, equally weighted catalogue types, paired across methods',
       'seed_marks':'Small gray points are the 10 training-seed means, not extra independent test worlds',
       'source_files':['analysis/test/paper_table.csv','analysis/test/seed_results.csv','analysis/test/results_summary.json'],
       'export':'183 mm by 105 mm; PDF, SVG, 300 dpi PNG; internal research report, no journal-specific compliance claim'
    }
    (FIG/'figure_contract.json').write_text(json.dumps(contract,indent=2),encoding='utf-8')
    plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
       'font.size':8,'axes.titlesize':9,'axes.labelsize':8,'xtick.labelsize':7,
       'ytick.labelsize':8,'pdf.fonttype':42,'ps.fonttype':42,'svg.fonttype':'none',
       'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,
       'savefig.dpi':300,'figure.dpi':150})
    fig,axes=plt.subplots(1,2,figsize=(183/25.4,105/25.4),sharey=True)
    fig.subplots_adjust(left=.27,right=.97,bottom=.22,top=.86,wspace=.28)
    y=np.arange(len(methods))[::-1]
    export_rows=[]
    for index,(method,pos) in enumerate(zip(methods,y)):
        row=rows[method]
        service=summaries['methods'][method]['groups']['all']['service_slack_vs_original']
        color='#0072B2' if method=='belief' else '#555555'
        for panel,keys in enumerate((('delta_revenue_vs_fixed','delta_ci_low','delta_ci_high'),
                                     ('estimate','ci95_low','ci95_high'))):
            source=row if panel==0 else service
            value,low,high=(float(source[key]) for key in keys)
            assert low<=high and np.isfinite([value,low,high]).all()
            # Percentile intervals can exclude a point estimate in unusual data;
            # use explicit segments rather than assuming symmetric errors.
            axes[panel].plot([low,high],[pos,pos],color=color,lw=1.6,zorder=3)
            axes[panel].scatter([value],[pos],s=26,c=color,zorder=4)
            subset=[r for r in seeds if r['method']==method and r['group']=='all']
            if len(subset)==10:
                metric='revenue_difference_vs_selected_fixed' if panel==0 else 'service_slack_vs_original'
                axes[panel].scatter([float(r[metric]) for r in subset],
                    pos+np.linspace(-.15,.15,10),s=8,color='#AAAAAA',alpha=.7,zorder=2)
            export_rows.append({'method':method,'panel':'Revenue' if panel==0 else 'Service',
                                'mean':value,'ci95_low':low,'ci95_high':high})
    for panel,axis in enumerate(axes):
        axis.axvline(0,color='#444444',lw=.8,linestyle=(0,(3,3)),zorder=1)
        axis.set_ylim(-.65,len(methods)-.35)
        axis.set_yticks(y,names)
        axis.tick_params(axis='y',length=0,pad=8)
        axis.grid(axis='x',color='#E5E5E5',lw=.5)
        axis.set_axisbelow(True)
        axis.set_title(['Revenue gain','Service margin'][panel],loc='left',pad=13)
        axis.text(-.1,1.08,'ab'[panel],transform=axis.transAxes,fontweight='bold',fontsize=9,
                  ha='left',va='baseline')
        axis.set_label('ab'[panel])
    axes[0].set_xlabel('Versus fixed threshold\n(normalized revenue units)',labelpad=7)
    axes[1].set_xlabel('Versus 95% baseline volume\n(core-hours)',labelpad=7)
    fig.text(.27,.06,'400 rounds; 16 fixed market types; 10 training seeds per GRU',fontsize=7)
    fig.text(.27,.025,'Dots: means. Lines: 95% paired intervals. Gray dots: seed means.',fontsize=7)
    fig.canvas.draw()
    require_two_panel_alignment(fig,axes,FIG/'threshold_comparison.alignment.json')
    fig.savefig(FIG/'threshold_comparison.pdf',facecolor='white')
    fig.savefig(FIG/'threshold_comparison.svg',facecolor='white')
    fig.savefig(FIG/'threshold_comparison.png',facecolor='white',dpi=300)
    plt.close(fig)
    with (FIG/'source_data.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(export_rows[0]));writer.writeheader();writer.writerows(export_rows)
    print(FIG/'threshold_comparison.png')
