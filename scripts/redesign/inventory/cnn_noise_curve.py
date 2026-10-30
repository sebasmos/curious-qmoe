"""Static macro F1 of the small CNNs against noise: FP32, simulated W8A8, real int8 with transplanted heads (way a)
and with retrained heads (way b). Source: stage 3c analysis/<ds>/static.csv (mean over seed and fold)."""
import pandas as pd, os, re
R=os.path.expanduser('~/Documents/qmoe-results/')
conds=['clean','bg__10db','bg__0db','white__10db','white__0db','field']
for ds in ['esc50','esc5','urban']:
    s=pd.read_csv(R+f'2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1/analysis/{ds}/static.csv')
    s=s[s.system.str.startswith('cnn:')&s.condition.isin(conds)]
    t=s.groupby(['system','condition']).f1.mean().unstack('condition')[[c for c in conds if c in set(s.condition)]]
    t=t[t.index.str.contains(r'cnn:mn(04|10|20)(int8a|int8b)?:(fp32|w8a8|int8):(esc|4)$')]
    print('##',ds); print(t.round(3).to_string())
