"""Head precision (FP32, 8-bit, 4-bit, BitNet) on the AudioSet CNN embeddings used in draft 06, W8A8 backbone.
Source: stage 3c analysis/<ds>/static.csv. Mean macro F1 over seed and fold."""
import pandas as pd, os
R=os.path.expanduser('~/Documents/qmoe-results/')
conds=['clean','bg__10db','bg__0db','white__10db','white__0db','field']
for ds in ['esc50','esc5','urban']:
    s=pd.read_csv(R+f'2026-10-23_1227_redesign-idea6-stage3c_code-9c9cbf1/analysis/{ds}/static.csv')
    s=s[s.system.str.match(r'^cnn:(mn10|mn20):w8a8:')&s.condition.isin(conds)]
    t=s.groupby(['system','condition']).f1.mean().unstack('condition')[conds]
    print('##',ds); print(t.round(3).to_string())
