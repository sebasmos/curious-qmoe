"""Heads on EfficientNet embeddings: per model per dataset macro F1 and paired difference to FP32.
Reads <run>/outputs/<ds>_singles_s<seed>/<kind>/fold_k/metrics.json. Pairs are (seed, fold).
CI: percentile bootstrap over pairs (10000, seed 0). p: paired t-test over (seed, fold) pairs and
over seed means (the more honest unit, small n)."""
import json, glob, sys, os, numpy as np
from scipy import stats
R=os.path.expanduser('~/Documents/qmoe-results/')
RUNS={'v3':R+'2026-10-20_2127_protocol-v3-singles_code-bc4ff1c/outputs','v2':R+'2026-10-20_1300_protocol-v2_code-5db7e75/outputs'}
KINDS=['1','2','4','8','16','bitnet','esc','qesc']
NAME={'1':'Q1','2':'Q2','4':'Q4','8':'Q8','16':'Q16(fp16)','bitnet':'BitNet','esc':'FP32','qesc':'FP32+PTQ int8'}
out=[]
for run,root in RUNS.items():
    for ds in ['esc','urban','quinn']:
        data={}; meta={}
        for k in KINDS:
            for f in glob.glob(f'{root}/{ds}_singles_s*/{k}/fold_*/metrics.json'):
                seed=int(f.split('_singles_s')[1].split('/')[0]); fold=int(f.split('fold_')[1].split('/')[0])
                m=json.load(open(f)); data.setdefault(k,{})[(seed,fold)]=m['f1_macro']
                meta[k]=(m['parameter_count'],m['model_size_bytes'])
        if 'esc' not in data: continue
        fp=data['esc']; rng=np.random.default_rng(0)
        for k in KINDS:
            if k not in data: continue
            d=data[k]; keys=sorted(set(d)&set(fp))
            seeds=sorted({s for s,_ in keys})
            sm=np.array([np.mean([d[(s,f)] for s2,f in keys if s2==s]) for s in seeds])
            row=dict(run=run,ds=ds,model=NAME[k],n_seeds=len(seeds),n_pairs=len(keys),
                f1_mean=float(np.mean([d[x] for x in keys])),sd_seeds=float(sm.std(ddof=1)) if len(sm)>1 else None,
                sd_pairs=float(np.std([d[x] for x in keys],ddof=1)),params=meta[k][0],nominal_bytes=meta[k][1])
            if k!='esc':
                diff=np.array([d[x]-fp[x] for x in keys])
                bs=[rng.choice(diff,len(diff)).mean() for _ in range(10000)]
                dsm=np.array([np.mean([d[(s,f)]-fp[(s,f)] for s2,f in keys if s2==s]) for s in seeds])
                row.update(diff=float(diff.mean()),ci=(float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))),
                    p_pairs=float(stats.ttest_1samp(diff,0).pvalue),p_seeds=float(stats.ttest_1samp(dsm,0).pvalue) if len(dsm)>1 else None)
            out.append(row)
json.dump(out,open('heads_table.json','w'),indent=1)
print('run ds model n_seeds n_pairs F1 sd_seed sd_pair | diff vs FP32 [95% CI] p_pairs p_seeds | params nominalKB')
for r in out:
    dd=f"{r['diff']:+.4f} [{r['ci'][0]:+.4f},{r['ci'][1]:+.4f}] {r['p_pairs']:.3g} {r['p_seeds'] if r['p_seeds'] is None else round(r['p_seeds'],3)}" if 'diff' in r else '-'
    print(r['run'],r['ds'],r['model'],r['n_seeds'],r['n_pairs'],f"{r['f1_mean']:.4f}",None if r['sd_seeds'] is None else round(r['sd_seeds'],4),round(r['sd_pairs'],4),'|',dd,'|',r['params'],round(r['nominal_bytes']/1024))
