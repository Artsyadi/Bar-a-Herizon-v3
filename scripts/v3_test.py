"""v3 on the held-out test match J03WPY (5 ball-degraded copies), next to v2, after v3 was chosen.

v3 was frozen and chosen for submission before this script ran; its result is reported as it is
and changes nothing (technical document, section 7). J03WPY was never used to train or choose.
Same frozen models and the same function that writes the BAR-ATH file (bh/v3post.py).

    BH_MODEL_VERSION=v2 python scripts/v3_test.py

Output: runs/v3/test_J03WPY.csv. If it exists, it is shown and nothing is re-scored.
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
V3 = os.path.join(config.PROJECT, "models", "v3")
OUT = os.path.join(config.PROJECT, "runs", "v3", "test_J03WPY.csv")


def check(manifest):
    man = json.load(open(manifest, encoding="utf-8"))
    bad = [rel for rel, h in man["md5"].items()
           if hashlib.md5(open(os.path.join(config.ROOT, rel), "rb").read()).hexdigest() != h]
    if bad:
        raise SystemExit("files changed since the freeze: {}".format(bad))
    return man


def show(df):
    pd.set_option("display.width", 200)
    print(df.pivot(index="unit", columns="system", values=["spellF1", "macroF1"]).round(4).to_string())
    s = df.groupby("system").agg(spellF1=("spellF1", "mean"), spellF1_worst=("spellF1", "min"),
                                 macroF1=("macroF1", "mean"), alive_dead_f1=("alive_dead_f1", "mean"),
                                 team_acc=("team_acc", "mean"), pred_spells=("pred_spells", "mean"),
                                 true_spells=("true_spells", "mean"))
    print(s.round(4).to_string())


def main():
    if os.environ.get("BH_MODEL_VERSION") != "v2":
        raise SystemExit("set BH_MODEL_VERSION=v2 (E1 features on, models/v2)")
    if os.path.isfile(OUT):
        print("already scored; showing the stored result (nothing re-scored)")
        show(pd.read_csv(OUT))
        return
    seg = check(os.path.join(V2, "manifest.json"))["segmenter"]
    check(os.path.join(V3, "manifest.json"))
    d_model = lgb.Booster(model_file=os.path.join(V2, "d_model.txt"))
    ha_model = lgb.Booster(model_file=os.path.join(V2, "ha_model.txt"))
    rows = []
    for m in config.split()["test_matches"]:
        fd, _ = d_features(m)
        frames, gt = load_labels(m)
        per = pd.read_parquet(config.path("raw_frames", match=m), columns=["frame", "period"]).set_index("frame") \
            .loc[frames, "period"].fillna(0).to_numpy()
        assert np.array_equal(fd.index.to_numpy(), frames)
        p_d = predict(d_model, fd)
        players = load_players(m)
        for mm, s in config.units(m):
            fh, _ = ha_features(mm, s)
            p_h = predict(ha_model, fh)
            ball = load_ball(mm, s)
            assert np.array_equal(players.frame, frames) and np.array_equal(ball.frame, frames)
            v2 = segment(p_d, p_h, per, seg["cost_d"], seg["cost_h"], seg["prob_clip"])
            v3 = v3post.labels(p_d, p_h, per, seg, players, ball)
            for system, lab in (("v2", v2), ("v3", v3)):
                r = score(gt, lab, per)
                rows.append({"unit": config.unit_id(mm, s), "system": system, "spellF1": r["spell_f1_mean"],
                             "macroF1": r["macro_f1"], "alive_dead_f1": r["alive_dead_f1"],
                             "team_acc": r["team_acc_gt_alive"], "pred_spells": r["n_pred_spells"],
                             "true_spells": r["n_gt_spells"]})
            print("{} scored".format(config.unit_id(mm, s)), flush=True)
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.to_csv(OUT, index=False)
    show(df)


if __name__ == "__main__":
    main()
