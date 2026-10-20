"""The run matrices emitted by scripts/reproduce_paper.sh --list and --list-extra.

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


# ── reviewer experiments: --list-extra ─────────────────────────────────────

def extra_lines():
    env = dict(os.environ, ESC_CSV="/data/esc.csv", QUINN_CSV="/data/quinn.csv", URBAN_CSV="/data/urban.csv",
               AST_ESC_CSV="/data/ast_esc.csv", AST_URBAN_CSV="/data/ast_urban.csv")
    out = subprocess.run(["bash", str(ROOT / "scripts" / "reproduce_paper.sh"), "--list-extra"], env=env,
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def kind(line):
    return re.sub(r"['\\]", "", re.search(r"models_to_run=\[(.*?)\]", line).group(1))


def base(line):
    return re.sub(r"_s\d+$", "", tag(line))


def test_extra_list_is_separate_from_the_paper_matrix():
    main, extra = job_lines(), extra_lines()
    assert len(main) == 202 and len(extra) == 217
    assert not {tag(l) for l in main} & {tag(l) for l in extra}, "no extra run reuses a paper tag"
    assert len({(tag(l), kind(l)) for l in extra}) == len(extra), "every extra run is a distinct job"
    for line in extra:
        assert "OMP_NUM_THREADS=2" in line and "experiment.num_threads=2" in line


def test_extra_seeds_datasets_and_label_column():
    for line in extra_lines():
        b, seed = base(line), int(re.search(r"experiment\.seed=(\d+)", line).group(1))
        ds = re.match(r"(ast_esc50c|ast_esc|ast_urban|esc50c|esc|quinn|urban)_", b).group(1)
        assert seed in (range(5) if ds in ("esc", "esc50c", "ast_esc", "ast_esc50c") else range(3))
        csv = {"esc": "esc", "esc50c": "esc", "quinn": "quinn", "urban": "urban",
               "ast_esc": "ast_esc", "ast_esc50c": "ast_esc", "ast_urban": "ast_urban"}[ds]
        assert f"experiment.datasets.esc.csv=/data/{csv}.csv" in line
        assert ("experiment.datasets.esc.label_col=label" in line) == ds.endswith("50c")


def test_extra_controls_and_arms():
    lines = extra_lines()
    kinds = Counter((base(l), kind(l)) for l in lines)
    for ds, n in (("esc", 5), ("quinn", 3), ("urban", 3)):
        assert kinds[(f"{ds}_singles_mp", "mp8_4")] == n
        for k in ("pm8_moe", "pm32_moe", "pm8_st", "pm32_st"):
            assert kinds[(f"{ds}_singles_pm", k)] == n
    for line in lines:
        if "_singles_pm_s" in line:
            assert "experiment.router.expert_quantizations=[bitnet,'\\''4'\\'','\\''8'\\'']" in line
            assert "experiment.router.num_experts=3" in line
    for ds, n in (("esc50c", 5), ("ast_esc", 5), ("ast_esc50c", 5), ("ast_urban", 3)):
        for k in ("esc", "4", "8", "bitnet", "qesc"):
            assert kinds[(f"{ds}_singles", k)] == n
        for arm in ("uniform_ctrl", "pprior_a1.0", "sharedtrunk_a1.0", "sharedtrunk_uniform_ctrl"):
            assert kinds[(f"{ds}_{arm}", "moe")] == n
        ctrl = next(l for l in lines if tag(l) == f"{ds}_sharedtrunk_uniform_ctrl_s0")
        assert "curiosity_alpha=0.0" in ctrl and "shared_trunk=true" in ctrl and "top_k=1" in ctrl
