"""Regenerate the submission file from the organisers' data bundle with the frozen models.

    python scripts/make_submission.py                    # everything (~3 minutes): v2 models + v3 rules
    python scripts/make_submission.py --step clean       # 1/2: read and clean the tactical ATD
    python scripts/make_submission.py --step predict     # 2/2: features, models, segmenter, CSV
    python scripts/make_submission.py --post none        # without the v3 rules of config/v3_post.json (the v2 file)

Options:
    --data-dir   folder that holds tactical/ (default: $MTAH_DATA_DIR, else more-than-a-hack-2027/data)
    --out        output CSV (default: outputs/Barça Herizon.csv)

Nothing here trains or reads labels. It prints checks only. Intermediate files go to outputs/bar_ath/.
If the organisers' repository is present (more-than-a-hack-2027/), their validate_submission.py is run.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time

import lightgbm as lgb

os.environ.setdefault("BH_MODEL_VERSION", "v2")      # v2 (v1 + E1) is the default model of this repository
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import bar_ath, config, v3post  # noqa: E402
from bh.models import predict  # noqa: E402
from bh.segmenter import segment  # noqa: E402

V1 = os.path.join(config.PROJECT, "models", os.environ.get("BH_MODEL_VERSION", "v1"))
DEFAULT_OUT = os.path.join(config.ROOT, "outputs", "Barça Herizon.csv")
V3 = os.path.join(config.PROJECT, "models", "v3", "manifest.json")


def show(checks):
    for name, ok in checks.items():
        print("  {:<50} {}".format(name, "OK" if ok else "FAILED"))
    if not all(checks.values()):
        sys.exit(1)


def verify_manifest():
    man = json.load(open(os.path.join(V1, "manifest.json"), encoding="utf-8"))
    changed = []
    for rel, h in man["md5"].items():
        with open(os.path.join(config.ROOT, rel), "rb") as f:
            if hashlib.md5(f.read()).hexdigest() != h:
                changed.append(rel)
    for rel in changed:
        print("  changed since the models were frozen:", rel)
    show({"frozen models, configs and code unchanged ({} files)".format(len(man["md5"])): not changed})
    return man


def verify_v3():
    """v3 = v2 + config/v3_post.json rules: the v2 manifest above, plus the v3 files unchanged."""
    man = json.load(open(V3, encoding="utf-8"))
    changed = [rel for rel, h in man["md5"].items()
               if hashlib.md5(open(os.path.join(config.ROOT, rel), "rb").read()).hexdigest() != h]
    for rel in changed:
        print("  changed since v3 was frozen:", rel)
    show({"v3 rules and code unchanged ({} files, base {})".format(len(man["md5"]), man["base"]):
          not changed and os.path.basename(V1) == man["base"]})


def step_clean(data_dir):
    t = time.time()
    print("1/2 read and clean the tactical ATD")
    show(bar_ath.clean(data=data_dir))
    print("    done in {:.0f} s".format(time.time() - t))


def step_predict(out, post="none"):
    t = time.time()
    print("2/2 features, frozen models, segmenter" + (", v3 rules" if post == "v3" else ""))
    man = verify_manifest()
    if post == "v3":
        verify_v3()
    seg = man["segmenter"]
    period, fd, fh = bar_ath.features()
    p_d = predict(lgb.Booster(model_file=os.path.join(V1, "d_model.txt")), fd)
    p_h = predict(lgb.Booster(model_file=os.path.join(V1, "ha_model.txt")), fh)
    if post == "v3":
        labels = v3post.labels(p_d, p_h, period, seg, bar_ath.load_players(), bar_ath.load_ball())
    else:
        labels = segment(p_d, p_h, period, seg["cost_d"], seg["cost_h"], seg["prob_clip"])
    bar_ath.write_submission(labels, out)
    checks = {"submission written: 199,319 rows, H/A/D": os.path.isfile(out)}
    validator = os.path.join(config.path("scorer_dir"), "validate_submission.py")
    if os.path.isfile(validator):
        r = subprocess.run([sys.executable, validator, out], capture_output=True, text=True)
        checks["organisers' validate_submission.py"] = r.returncode == 0
    else:
        print("  (organisers' repository not found: run validate_submission.py yourself)")
    show(checks)
    with open(out, "rb") as f:
        print("    md5 {}".format(hashlib.md5(f.read()).hexdigest()))
    print("    done in {:.0f} s -> {}".format(time.time() - t, out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--step", choices=["clean", "predict", "all"], default="all")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    ap.add_argument("--post", choices=["none", "v3"], default=os.environ.get("BH_POST", "v3"),
                    help="v3 (default): the two rules of config/v3_post.json on top of the v2 models; none: v2 alone")
    a = ap.parse_args()
    if a.step in ("clean", "all"):
        step_clean(a.data_dir)
    if a.step in ("predict", "all"):
        step_predict(a.out, a.post)


if __name__ == "__main__":
    main()
