# Figure scripts

The scripts that draw the three data figures of the paper, copied from the paper's figure folder; Figure 1 is a
schematic with no measured number and is not here. Each script reads stored run folders under `RESULTS_DIR`, asserts
that what it draws matches the run's tables, and writes `<name>.pdf`, `.png` and `.json` (every plotted number with its
source path) next to itself. `reference/` holds the JSON files of the figures in the submitted PDF, for comparison.
Matplotlib only, CPU, a few seconds each: `cd scripts/figures && python fig-humility.py`.

| Figure | Script | Reads |
|---|---|---|
| `fig:t21` | `fig-t21-fresh-data-v2.py` | T21 `tables/d1/`, `tables/d2/`; T25 `tables/d1/` |
| `fig:humility` | `fig-humility.py` | T3/T5 `tables/t5_rate_by_cell_q30.csv`; T21 `tables/{d1,d2}/primary_by_condition.csv` |
| `fig:gateprec` | `fig-gate-by-precision-v2.py` | gate by precision `paired_vs_fp32.csv`, `systems.csv` |

`figstyle.py`, `fig_v2_common.py` (label overlap check) and `fig_v3_common.py` (size, font and contrast asserts) are
shared helpers.
