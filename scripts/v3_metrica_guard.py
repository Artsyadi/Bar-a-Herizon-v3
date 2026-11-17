"""The one final look for v3: Metrica only (never J03WPY, already used for v2). Run ONCE.

v2 models (models/v2, manifest checked) on the 3 Metrica matches, with and without the v3 rules.
Check fixed 15 Nov before any v3 result: v3's Metrica mean spellF1 >= 0.7001 (v2 0.7201 - 0.02).
A collapse guard only: Metrica is never used to choose anything.

    BH_MODEL_VERSION=v2 python scripts/v3_metrica_guard.py

Needs LightGBM, the cleaned Metrica data (data/cleaned) and the raw Metrica frames and labels.
Output: runs/v3/metrica_guard.csv. If it exists, it is shown and nothing is re-scored.
"""

import hashlib
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config, v3post  # noqa: E402
from bh.data import load_ball, load_players  # noqa: E402
from bh.models import predict  # noqa: E402
from bh.pipeline import d_features, ha_features  # noqa: E402
from bh.scoring import load_labels, score  # noqa: E402
from bh.segmenter import segment  # noqa: E402

V2 = os.path.join(config.PROJECT, "models", "v2")
OUT = os.path.join(config.PROJECT, "runs", "v3", "metrica_guard.csv")
THRESHOLD = 0.7001


def show(df):
    r = df.groupby("system").spellF1.mean()
    print(df.pivot(index="match", columns="system", values="spellF1").round(4).to_string())
    print("Metrica mean spellF1: v2 {:.4f}, v3 {:.4f} -> check v3 >= {}: {}".format(
        r["v2"], r["v3"], THRESHOLD, "PASS" if r["v3"] >= THRESHOLD else "FAIL"))


def main():
    if os.environ.get("BH_MODEL_VERSION") != "v2":
        raise SystemExit("set BH_MODEL_VERSION=v2 (E1 features on, models/v2)")
    if os.path.isfile(OUT):
        print("already scored once; showing the stored result (nothing re-scored)")
        show(pd.read_csv(OUT))
        return
    man = json.load(open(os.path.join(V2, "manifest.json"), encoding="utf-8"))
    bad = [rel for rel, h in man["md5"].items()
           if hashlib.md5(open(os.path.join(config.ROOT, rel), "rb").read()).hexdigest() != h]
    if bad:
        raise SystemExit("v2 files changed: {}".format(bad))
    seg = man["segmenter"]
    d_model = lgb.Booster(model_file=os.path.join(V2, "d_model.txt"))
    ha_model = lgb.Booster(model_file=os.path.join(V2, "ha_model.txt"))
    rows = []
    for m in config.split()["comparison_matches"]:
        fd, _ = d_features(m)
        frames, gt = load_labels(m)
        per = pd.read_parquet(config.path("raw_frames", match=m), columns=["frame", "period"]).set_index("frame") \
            .loc[frames, "period"].fillna(0).to_numpy()
        assert np.array_equal(fd.index.to_numpy(), frames)
        p_d = predict(d_model, fd)
        fh, _ = ha_features(m, None)
        p_h = predict(ha_model, fh)
        p, b = load_players(m), load_ball(m, None)
        assert np.array_equal(p.frame, frames) and np.array_equal(b.frame, frames)
        v2 = segment(p_d, p_h, per, seg["cost_d"], seg["cost_h"], seg["prob_clip"])
        v3 = v3post.labels(p_d, p_h, per, seg, p, b)
        for system, lab in (("v2", v2), ("v3", v3)):
            rows.append({"match": m, "system": system, "spellF1": score(gt, lab, per)["spell_f1_mean"]})
        print("{} scored".format(m), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    show(df)


if __name__ == "__main__":
    main()
