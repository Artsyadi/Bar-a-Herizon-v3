"""v3 rule tests (bh/v3post.py) on made-up data. No match data needed.

    python tests/test_v3post.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bh.data import BallData, PlayerData  # noqa: E402
from bh import v3post  # noqa: E402

PITCH = (105.0, 68.0)
SEG = {"cost_d": 64, "cost_h": 8, "prob_clip": 1e-4}


def lab(s):
    return np.array(list(s))


def ball(xy, status, period=None):
    n = len(xy)
    return BallData("t", np.arange(n), np.ones(n, np.int16) if period is None else np.asarray(period),
                    np.asarray(xy, np.float32), np.asarray(status, np.int8), np.ones(n, np.float32))


def players(n, xy=None):
    xy = np.full((n, 2, 2), np.nan, np.float32) if xy is None else xy
    return PlayerData("t", np.arange(n), np.ones(n, np.int16), xy, np.zeros((n, 2), np.float32),
                      np.array(["H", "A"]), np.array([0, 0]), PITCH, 25.0)


def test_ball_out_lines_margin_status_and_halves():
    xy = [[0, 0], [52.9, 0], [53.1, 0], [-53.1, 0], [0, 34.4], [0, -34.6], [60, 0], [60, 0]]
    b = ball(xy, [0, 0, 0, 0, 0, 0, 1, 0], [1, 1, 1, 1, 1, 1, 1, 0])
    assert v3post.ball_out(b, PITCH, 0.5).tolist() == [False, False, True, True, False, True, False, False]


def test_ball_out_uses_the_match_pitch():
    b = ball([[52.85, 0]], [0])
    assert v3post.ball_out(b, (105.0, 68.0), 0.25).tolist() == [True]
    assert v3post.ball_out(b, (105.3, 68.0), 0.25).tolist() == [False]


def test_touch_candidates():
    per = np.ones(12, int)
    th, ta = np.zeros(12, bool), np.zeros(12, bool)
    ta[3], th[7] = True, True
    assert "".join(v3post.touch_candidates(lab("DHHHHHHHHHHD"), per, th, ta, 50, 3)) == "DHHAAAAHHHHD"
    assert "".join(v3post.touch_candidates(lab("DHHHHHHHHHHD"), per, th, ta, 50, 5)) == "DHHHHHHHHHHD"
    th, ta = np.zeros(12, bool), np.zeros(12, bool)
    ta[2] = True
    assert "".join(v3post.touch_candidates(lab("HHHHHHHHHHHH"), per, th, ta, 4, 1)) == "HHAAAAHHHHHH"
    assert "".join(v3post.touch_candidates(lab("HHHHHDHHHHHH"), per, th, ta, 50, 1)) == "HHAAADHHHHHH"
    per2 = np.array([1] * 4 + [2] * 8)
    assert "".join(v3post.touch_candidates(lab("HHHHHHHHHHHH"), per2, th, ta, 50, 1)) == "HHAAHHHHHHHH"


def test_ball_out_turns_a_live_stretch_dead():
    n = 300
    xy = np.zeros((n, 2))
    xy[100:200, 0] = 54.0                                  # ball observed out for 4 s
    b = ball(xy, np.zeros(n))
    out = v3post.labels(np.full(n, 0.01), np.full(n, 0.95), np.ones(n), SEG, players(n), b)
    assert set(out[100:200]) == {"D"} and set(out[:90]) == {"H"} and set(out[210:]) == {"H"}
    no_rule = v3post.labels(np.full(n, 0.01), np.full(n, 0.95), np.ones(n), SEG, players(n), ball(np.zeros((n, 2)), np.zeros(n)))
    assert set(no_rule) == {"H"}                           # no touches, ball in: v2's labels


def test_touch_detector_sees_a_kick_next_to_an_away_player():
    n = 100
    xy = np.zeros((n, 2))
    xy[:50, 0] = np.arange(50) * 0.2                       # 5 m/s to the right
    xy[50:, 0] = xy[49, 0] - np.arange(1, 51) * 0.2        # kicked back at 5 m/s at frame 50
    pl = np.full((n, 2, 2), 50.0, np.float32)              # home player far away
    pl[:, 1, :] = xy.astype(np.float32) + np.float32(1.0)   # away player 1.4 m from the ball
    t = v3post.detect_touches(players(n, pl), ball(xy, np.zeros(n)), 1.5, 3.0, 0.4)
    assert np.flatnonzero(t).tolist() == [49] and t[49] == -1


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
    print("v3post: {} tests passed".format(len(tests)))
