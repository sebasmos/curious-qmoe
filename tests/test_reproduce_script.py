"""The run matrix emitted by scripts/reproduce_paper.sh --list.

Run: PYTHONPATH=. pytest tests/test_reproduce_script.py -q
"""
import os
import re
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def job_lines():
    env = dict(os.environ, ESC_CSV="/data/esc.csv", QUINN_CSV="/data/quinn.csv", URBAN_CSV="/data/urban.csv")
    out = subprocess.run(["bash", str(ROOT / "scripts" / "reproduce_paper.sh"), "--list"], env=env,
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def tag(line):
    return re.search(r"experiment\.metadata\.tag=(\S+)", line).group(1)


def test_seeds_per_dataset():
    seeds = {}
    for line in job_lines():
        ds = re.search(r"--config-name (\S+)", line).group(1)
        seeds.setdefault(ds, set()).add(int(re.search(r"experiment\.seed=(\d+)", line).group(1)))
    assert seeds == {"esc50": {0, 1, 2, 3, 4}, "quinn": {0, 1, 2}, "urban8k": {0, 1, 2}}


def test_every_run_is_tagged_with_its_seed_and_runs_once():
    lines = job_lines()
    for line in lines:
        seed = re.search(r"experiment\.seed=(\d+)", line).group(1)
        assert tag(line).endswith(f"_s{seed}")
    moe = Counter(tag(l) for l in lines if "models_to_run=[moe]" in l)
    assert max(moe.values()) == 1, "no MoE run is duplicated"
    assert not any("_r2" in tag(l) for l in lines), "seeds replace the old replication runs"


def test_uniform_controls_are_the_bayesian_router_at_alpha_zero():
    lines = job_lines()
    for ds in ("esc", "quinn", "urban"):
        for name in (f"{ds}_uniform_ctrl_s0", f"{ds}_sharedtrunk_uniform_ctrl_s0"):
            line = next(l for l in lines if tag(l) == name)
            assert "experiment.router.use_curiosity=true" in line
            assert "experiment.router.curiosity_alpha=0.0" in line
        assert "shared_trunk=true" in next(l for l in lines if tag(l) == f"{ds}_sharedtrunk_uniform_ctrl_s0")


def test_jobs_pin_threads_route_top1_and_skip_popcount():
    for line in job_lines():
        assert "OMP_NUM_THREADS=2" in line and "experiment.num_threads=2" in line
        assert "experiment.track_emissions=false" in line
        assert "popcount" not in line
        if "models_to_run=[moe]" in line:
            assert "experiment.router.top_k=1" in line
