"""v3: the frozen v2 models and segmenter plus two physics rules (config/v3_post.json).

    labels = v3post.labels(p_d, p_h, period, segmenter_cfg, players, ball)

1. Ball out of play: frames whose observed ball is past a line get p_d = 1 before the segmenter.
2. Touch candidates: after the segmenter, an opponent's touch inside a predicted possession opens a
   short possession for the toucher, until the next touch of the other team (at most 2 s).

The same function makes the BAR-ATH file (scripts/make_submission.py --post v3) and the dev check
(scripts/v3_post_check.py), so what was measured is what is submitted. Nothing here is learned.
"""

import json
import os

import numpy as np

from . import config
from .data import config_features
from .features_common import lag_values, segments
from .features_ha import _local_extremum
from .segmenter import segment


def settings():
    with open(os.path.join(config.CONFIG, "v3_post.json"), encoding="utf-8") as fh:
        return json.load(fh)


def ball_out(ball, pitch, margin_m):
    """True where the observed ball is more than margin_m beyond a touchline or goal line, inside a
    half. Coordinates are centred on the pitch (metres)."""
    x, y = ball.xy[:, 0].astype(np.float64), ball.xy[:, 1].astype(np.float64)
    half_l, half_w = pitch[0] / 2.0, pitch[1] / 2.0
    with np.errstate(invalid="ignore"):
        beyond = (np.abs(x) > half_l + margin_m) | (np.abs(y) > half_w + margin_m)
    return (ball.status == 0) & beyond & (ball.period > 0)


def detect_touches(players, ball, touch_radius_m, touch_dv_min_mps, lag_s):
    """Per frame: +1 home touch, -1 away touch, 0 none. E1's detector (bh/features_ha.py,
    ha_features_v2): the ball velocity changes by at least touch_dv_min_mps between lag_s before
    and after (local maximum), the ball observed at t - lag_s, t and t + lag_s, a player within
    touch_radius_m of the ball; the nearer team touches."""
    if not np.array_equal(players.frame, ball.frame):
        raise ValueError("player and ball frames differ")
    fps = players.fps
    segs = segments(players.period)
    b = ball.xy.astype(np.float32)
    k = int(round(lag_s * fps))
    vin = (b - lag_values(b, segs, k)) / (k / fps)
    vout = (lag_values(b, segs, -k) - b) / (k / fps)
    observed = ball.status == 0
    obs_f = observed.astype(np.float32)
    ok = observed & (lag_values(obs_f, segs, k) == 1) & (lag_values(obs_f, segs, -k) == 1)
    dv = np.where(ok, np.hypot(*(vout - vin).T), np.nan)
    xy, team = players.xy, players.team
    d = np.hypot(xy[..., 0] - b[:, None, 0], xy[..., 1] - b[:, None, 1])
    near = {}
    for t, m in (("h", team == "H"), ("a", team == "A")):
        dt = np.where(m[None, :], d, np.nan)
        allnan = ~np.isfinite(dt).any(axis=1)
        near[t] = np.where(allnan, np.nan, np.nanmin(np.where(allnan[:, None], 0, dt), axis=1))
    with np.errstate(invalid="ignore"):
        h_near = near["h"] <= touch_radius_m
        a_near = near["a"] <= touch_radius_m
        who = np.where(h_near & (~a_near | (near["h"] <= near["a"])), 1, np.where(a_near, -1, 0))
        hit = ok & (dv >= touch_dv_min_mps) & _local_extremum(dv, segs, k, "max")
    return np.where(hit, who, 0).astype(np.int8)


def touch_candidates(labels, periods, touch_h, touch_a, max_len, min_len):
    """Inside a live H/A stretch predicted for team X, a touch of the other team Y relabels frames
    from the touch to Y's opponent's next touch as Y, at most max_len frames, stopping at a non-live
    frame or a period change; only if >= min_len frames. Candidates are found on the input labels."""
    lab = np.asarray(labels).astype(str)
    per = np.asarray(periods)
    out = lab.copy()
    live = np.isin(lab, ["H", "A"])
    idx = {"H": np.flatnonzero(touch_h), "A": np.flatnonzero(touch_a)}
    n = len(lab)
    for team, other in (("H", "A"), ("A", "H")):
        nxt = idx[other]
        for f in idx[team]:
            if not live[f] or lab[f] == team:
                continue
            j = np.searchsorted(nxt, f, side="right")
            g = min(nxt[j] if j < len(nxt) else n, f + max_len)
            e = f
            while e < g and live[e] and per[e] == per[f]:
                e += 1
            if e - f >= min_len:
                out[f:e] = team
    return out


def labels(p_d, p_h, period, seg, players, ball, cfg=None):
    """H/A/D per frame: v2's segmenter with the two v3 rules. seg = {cost_d, cost_h, prob_clip}."""
    cfg = cfg or settings()
    fc = config_features()
    period = np.asarray(period)
    if not (len(p_d) == len(p_h) == len(period) == len(ball.frame) == len(players.frame)):
        raise ValueError("probabilities, players and ball must cover the same frames")
    clip = seg["prob_clip"]
    q = np.asarray(p_d, dtype=np.float64).copy()
    q[ball_out(ball, players.pitch, cfg["ball_out"]["margin_m"])] = 1.0 - clip
    lab = np.asarray(segment(q, p_h, period, seg["cost_d"], seg["cost_h"], clip)).astype(str)
    e1 = fc["ha_features_v2"]
    t = detect_touches(players, ball, e1["touch_radius_m"], e1["touch_dv_min_mps"],
                       fc["ha_features"]["ball_speed_change_lag_s"])
    tc = cfg["touch_candidates"]
    return touch_candidates(lab, period, t == 1, t == -1, tc["max_len_frames"], tc["min_len_frames"])
