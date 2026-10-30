import pandas as pd, os
B=os.path.expanduser('~/Documents/qmoe-results/2026-10-20_2359_redesign-idea5-stage2_code-3c18c38/analysis/')
for d in ['esc50','esc5','urban']:
    s=pd.read_csv(B+d+'/static.csv'); s=s[s['head']=='esc']
    t=s[s.condition=='clean'].groupby('config').agg(f1=('f1','mean'),sd=('f1','std'),n=('f1','size'),bitops=('bitops','first'),stor=('storage_bytes','first'))
    t['bitops']=t.bitops/1e12; t['stor']=t.stor/1e6
    print('##',d); print(t.round(4).sort_values('bitops',ascending=False).to_string())
