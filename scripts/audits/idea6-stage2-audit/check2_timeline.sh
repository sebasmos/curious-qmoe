#!/bin/bash
# Check 2: pre-registration integrity. Read only: git log/reflog/diff of the worktree, file mtimes of the
# claim folder. Output: logs/check2_timeline.log
W=/Users/cajas.sebastian/Desktop/repositories/curious-qmoe-redesign-idea6-stage2
R=/Users/cajas.sebastian/Documents/qmoe-results/2026-10-22_1120_redesign-idea6-stage2_code-9f346ba
cd $W
echo "== commits (author and committer time)"; git log --format='%h %ad | %cd | %s' --date=format:'%H:%M:%S' 688a498..c892c2f | cut -c1-150
echo "== reflog"; git reflog --date=format:'%H:%M:%S' -n 14 | cut -c1-90
echo "== earliest score files (any condition) and earliest noisy ones"
find $R/scores $R/scores_ast_short $R/scores_ast1024 -name '*.npz' -print0 | xargs -0 stat -f '%Sm %N' -t '%H:%M:%S' | sort | head -3
find $R/scores $R/scores_ast_short $R/scores_ast1024 -name '*.npz' ! -name '*clean*' -print0 | xargs -0 stat -f '%Sm %N' -t '%H:%M:%S' | sort | head -3
echo "== files in the claim folder written between 12:03 and 15:46 outside scores/emb/outputs/logs"
find $R -type f -newermt '2026-10-22 12:03:00' ! -newermt '2026-10-22 15:46:00' ! -path '*/scores*' ! -path '*/emb*' ! -path '*/outputs/*' ! -path '*/logs/*' -print0 | xargs -0 stat -f '%Sm %N' -t '%H:%M:%S' | sort
echo "== what changed in analysis code after the first noisy score (fa462ce..c892c2f, analyze.py, score_eff.py, benchmark.py)"
git diff --stat fa462ce c892c2f
git diff fa462ce c892c2f -- scripts/redesign/idea6/stage2/analyze.py | grep '^[-+]' | grep -v '^+++\|^---'
echo "== plan text changes after the first noisy score"
git diff fa462ce c892c2f -- scripts/redesign/idea6/stage2/ANALYSIS_PLAN.md | grep '^[-+]' | grep -v '^+++\|^---' | cut -c1-160
echo "== plan text changes at amendment 1 and 2 (removed lines only: anything taken back?)"
git diff 14c8504 fa462ce -- scripts/redesign/idea6/stage2/ANALYSIS_PLAN.md | grep '^-' | grep -v '^---'
echo "== budget grid, hull and selection in analyze.py at fa462ce"
git show fa462ce:scripts/redesign/idea6/stage2/analyze.py | grep -n 'BUDGETS =\|RULES =\|EXPENSIVE_FULL =\|^SHORT =\|def hull_pair\|sort_values(\["gain_iv"' 
echo "== copy in the claim folder equals c892c2f"
for f in analyze.py score_eff.py summary.py; do git show c892c2f:scripts/redesign/idea6/stage2/$f | diff -q - $R/scripts/$f && echo "$f same"; done
echo "== first final pass and rerun: H1 rows"
diff <(cut -d, -f1-13 $R/analysis_first_pass_without_w4short/hypothesis_H1.csv) <(cut -d, -f1-13 $R/analysis/hypothesis_H1.csv) && echo "H1 identical in both passes (first 13 columns)"
diff $R/analysis_first_pass_without_w4short/hypothesis_H3.csv $R/analysis/hypothesis_H3.csv
