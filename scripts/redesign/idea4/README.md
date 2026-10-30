# Idea 4: learned router inside the QMoE head (negative result)

The learned router changed the library itself (`QWave/moe.py`, `scripts/benchmark.py`, the four configs), so its
code is kept here as a snapshot of the files as they were when the idea 4 stage 1 run was made, instead of being merged
into the main pipeline. The main tables are produced by the unmodified library at the repository root.

To rerun idea 4, copy the snapshot over a fresh unpacked copy of this archive and run the snapshot's own tests:

```
cp -R scripts/redesign/idea4/snapshot/. .
PYTHONPATH=. pytest tests/test_router_redesign.py
PYTHONPATH=. python scripts/benchmark.py --config-name esc50 experiment.router.dense_experts=true   # options off by default, keys under `router` in the config
PYTHONPATH=. python scripts/analysis/router_diagnostics.py --help
```

Do this in a separate copy: the snapshot's `benchmark.py` predates the protocol 3 grouped inner split used by the
main tables.

`summarize.py` (copied from the stage 1 run folder's `analysis/`) turns `diag_esc_s0.json` into the stage 1 summary and
`diag_esc_s0_verdict.json`: `python summarize.py diag_esc_s0.json [uniform_ctrl]`.
