"""v3 dev check: the exact function that makes the v3 BAR-ATH file, on the 30 IDSSE dev units.

External data only (6 dev matches x 5 ball seeds); never J03WPY, never BAR-ATH. Needs:
    runs/phase4/d/oof_<match>.parquet, runs/phase4/ha/oof_<unit>.parquet   (v2 out-of-fold probabilities)
    data/cleaned/frames/<unit>.parquet, data/cleaned/players/<match>.parquet   (step12_clean_all.py)
    data/features/train_ha/<unit>.parquet   (v2's touch columns, to check the touch detector)

    python scripts/v3_post_check.py

Prints, per unit and overall: the touch detector equals v2's E1 touches on every stored row; v2
reproduced (0.6135); v3 and the keep rule (gain > 0.016 and >= 5 of 6 matches better).
"""

import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh import config, v3post  # noqa: E402
from bh.data import config_features, load_ball, load_players  # noqa: E402
from bh.features_common import segments  # noqa: E402
from bh.features_ha import _frames_since  # noqa: E402
from bh.scoring import score  # noqa: E402
from bh.segmenter import segment  # noqa: E402

PHASE4 = os.path.join(config.PROJECT, "runs", "phase4")
TABLES = os.path.join(config.PROJECT, "data", "features", "train_ha")
V2_DEV = 0.6135


def main():
    seg = json.load(open(os.path.join(config.PROJECT, "models", "v2", "manifest.json"), encoding="utf-8"))["segmenter"]
    fc = config_features()
    e1 = fc["ha_features_v2"]
    cap = int(e1["since_cap_s"] * 25)
    rows, touch_ok = [], True
    for _, val in config.folds():
        d = pd.read_parquet(os.path.join(PHASE4, "d", "oof_{}.parquet".format(val)))
        p = load_players(val)
        per, gt = d.period.to_numpy(), d.label.to_numpy().astype(str)
        for m, s in config.units(val):
            uid = config.unit_id(m, s)
            h = pd.read_parquet(os.path.join(PHASE4, "ha", "oof_{}.parquet".format(uid)))
            b = load_ball(m, s)
            if not (np.array_equal(p.frame, d.frame.to_numpy()) and np.array_equal(b.frame, h.frame.to_numpy())):
                raise SystemExit("{}: frames differ between cleaned data and v2 outputs".format(uid))
            # the touch detector must be E1 exactly
            t = v3post.detect_touches(p, b, e1["touch_radius_m"], e1["touch_dv_min_mps"],
                                      fc["ha_features"]["ball_speed_change_lag_s"])
            tab = pd.read_parquet(os.path.join(TABLES, uid + ".parquet"), columns=["ha_touch_since_h_s", "ha_touch_since_a_s"])
            pos = pd.Series(np.arange(len(p.frame)), index=p.frame).loc[tab.index].to_numpy()
            for side, code in (("h", 1), ("a", -1)):
                mine = (_frames_since(t == code, segments(p.period), cap) / 25.0)[pos]
                theirs = tab["ha_touch_since_{}_s".format(side)].to_numpy()
                touch_ok &= bool((np.isclose(mine, theirs, atol=1e-4) | (np.isnan(mine) & np.isnan(theirs))).all())
            v2 = segment(d.p_d.to_numpy(), h.p_h.to_numpy(), per, seg["cost_d"], seg["cost_h"], seg["prob_clip"])
            v3 = v3post.labels(d.p_d.to_numpy(), h.p_h.to_numpy(), per, seg, p, b)
            rows.append(dict(match=val, unit=uid, v2=score(gt, v2, per)["spell_f1_mean"],
                             v3=score(gt, v3, per)["spell_f1_mean"]))
            print("  {:<18} v2 {:.4f}  v3 {:.4f}".format(uid, rows[-1]["v2"], rows[-1]["v3"]), flush=True)
    r = pd.DataFrame(rows).groupby("match")[["v2", "v3"]].mean()
    gain = r.v3 - r.v2
    print("\ntouch detector identical to v2's E1 on every stored row:", "PASS" if touch_ok else "FAIL")
    print("v2 reproduced: {:.4f} (expected {:.4f}) -> {}".format(r.v2.mean(), V2_DEV,
                                                               "PASS" if abs(r.v2.mean() - V2_DEV) < 5e-4 else "FAIL"))
    kept = gain.mean() > 0.016 and (gain > 0).sum() >= 5
    print("v3: {:.4f}, gain {:+.4f}, {}/6 matches better, worst {:+.4f} -> keep rule {}".format(
        r.v3.mean(), gain.mean(), int((gain > 0).sum()), gain.min(), "PASS" if kept else "FAIL"))


if __name__ == "__main__":
    main()
