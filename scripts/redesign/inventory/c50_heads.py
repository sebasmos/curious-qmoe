"""ESC-50 at 50 classes, heads on EfficientNet embeddings, seed 0 only (5 folds), protocol 3.
Source: 2026-10-20_2300_redesign-headroom2-esc50class-tinyhead_code-bc4ff1c/outputs_*"""
import json,glob,os,numpy as np
from scipy import stats
B=os.path.expanduser('~/Documents/qmoe-results/2026-10-20_2300_redesign-headroom2-esc50class-tinyhead_code-bc4ff1c/')
for d in ['outputs_c50','outputs_tiny5','outputs_tiny50']:
    fp={}
    for k in ['esc','1','2','4','8','bitnet']:
        fs=sorted(glob.glob(f'{B}{d}/*/{k}/fold_*/metrics.json'))
        v=np.array([json.load(open(f))['f1_macro'] for f in fs]); m=json.load(open(fs[0]))
        if k=='esc': fp=v
        dd='' if k=='esc' else f" diff {np.mean(v-fp):+.4f} p {stats.ttest_1samp(v-fp,0).pvalue:.2f}"
        print(d,k,len(v),f"F1 {v.mean():.4f} sd {v.std(ddof=1):.4f} params {m['parameter_count']} nominal {m['model_size_bytes']:.0f}B"+dd)
