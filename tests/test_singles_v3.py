"""The protocol 3 single-model jobs of scripts/redesign/singles_v3.sh --list:
the protocol 2 singles of scripts/reproduce_paper.sh --list, grouped inner
split on, outputs and logs under $V3_OUTPUTS.

Run: PYTHONPATH=. pytest tests/test_singles_v3.py -q
"""
import os
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CSVS = dict(ESC_CSV="/data/esc.csv", QUINN_CSV="/data/quinn.csv", URBAN_CSV="/data/urban.csv")


def lines(script, *args, **env):
    out = subprocess.run(["bash", str(ROOT / "scripts" / script), *args], env=dict(os.environ, **CSVS, **env),
                         capture_output=True, text=True, check=True).stdout
    return [line for line in out.splitlines() if line.strip()]


def test_one_job_per_dataset_seed_and_model(tmp_path):
    v3 = lines("redesign/singles_v3.sh", "--list", V3_OUTPUTS=str(tmp_path))
    keys = set()
    for line in v3:
        ds = re.search(r"--config-name (\S+)", line).group(1)
        seed = int(re.search(r"experiment\.seed=(\d+)", line).group(1))
        kind = re.search(r"models_to_run=\[(.*?)\]", line).group(1).replace("'", "").replace("\\", "")
        keys.add((ds, seed, kind))
        tag = re.search(r"experiment\.metadata\.tag=(\S+)", line).group(1)
        assert tag == {"esc50": "esc", "quinn": "quinn", "urban8k": "urban"}[ds] + f"_singles_s{seed}"
        assert "experiment.cross_validation.inner_split_groups=recording" in line
        assert f"'experiment.output_root={tmp_path}'" in line and f"'hydra.run.dir={tmp_path}/{tag}'" in line
        assert f"> '{tmp_path}/logs/" in line
    kinds = {"1", "2", "4", "8", "16", "esc", "qesc", "bitnet"}
    expected = {(d, s, k) for d, seeds in (("esc50", range(5)), ("quinn", range(3)), ("urban8k", range(3)))
                for s in seeds for k in kinds}
    assert keys == expected and len(v3) == len(expected) == 88
    assert (tmp_path / "logs").is_dir()


def test_same_recipe_as_the_protocol_2_singles(tmp_path):
    v2 = [l for l in lines("reproduce_paper.sh", "--list") if "_singles_s" in l]
    v3 = lines("redesign/singles_v3.sh", "--list", V3_OUTPUTS=str(tmp_path))
    strip = re.compile(r" experiment\.cross_validation\.inner_split_groups=recording"
                       r" 'experiment\.output_root=[^']*' 'hydra\.run\.dir=[^']*'")
    prefix = lambda l: l.split(" experiment.metadata.tag=")[0]
    assert sorted(prefix(strip.sub("", l)) for l in v3) == sorted(prefix(l) for l in v2)


def test_list_needs_the_outputs_dir():
    env = dict(os.environ, **CSVS)
    env.pop("V3_OUTPUTS", None)
    res = subprocess.run(["bash", str(ROOT / "scripts" / "redesign" / "singles_v3.sh"), "--list"], env=env,
                         capture_output=True, text=True)
    assert res.returncode != 0 and "V3_OUTPUTS" in res.stderr
