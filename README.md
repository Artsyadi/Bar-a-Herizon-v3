# Barça Herizon · More than a Hack 2027

Our Phase 1 entry: for every frame of the BAR-ATH tactical video, who has the ball — home (`H`), away
(`A`) or nobody because the ball is dead (`D`).

The system reads the provided tracking data (tactical ATD), cleans it, and combines two models with a
segmenter:

- a **dead-ball model** that decides dead vs live from the players' movement and shape only;
- a **possession model** that decides home vs away on live frames from players and ball, including
  evidence that reacts within a second (who touched the ball last, who is within a metre of it);
- a **two-layer segmenter** (Viterbi) that turns the frame probabilities into possessions;
- two **physics rules** around the segmenter: a ball seen beyond a touchline or goal line is dead,
  and an opponent's touch inside a possession opens a short possession for that team.

Both models were trained only on external matches (IDSSE). Nothing was trained, tuned or calibrated
on the provided game or its video. The approach, the data and the decisions behind them are in the
technical document, [`docs/Barça Herizon.md`](docs/Barça%20Herizon.md).

## Reproduce the submission file

Python 3.10, with the pinned versions in `requirements.txt` (the frozen models were produced with Python 3.10.20).

```bash
git clone https://github.com/Artsyadi/Bar-a-Herizon-v3.git
cd Bar-a-Herizon-v3
python3.10 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put the organisers' repository and data bundle inside the repository folder (both are ignored by
git and never committed):

```
Bar-a-Herizon-v3/
└── more-than-a-hack-2027/           # git clone of the organisers' repository
    └── data/tactical/               # the data bundle, as described in their DATA.md
        ├── BAR-ATH_tactical_atd.xml
        └── BAR-ATH_tactical_atd.txt
```

The data can also live elsewhere: set `MTAH_DATA_DIR` to the folder that holds `tactical/`, or pass
`--data-dir`.

Then:

```bash
python scripts/make_submission.py
```

This writes `outputs/Barça Herizon.csv` (about 3 minutes; intermediate files in `outputs/bar_ath/`)
and, when the organisers' repository is present, runs their `validate_submission.py` on it. The
file's md5 is `bc3a2a71ffae604b98e798ba843dbd67`.

The script first checks that the frozen models, configuration and code match the fingerprints in
`models/v2/manifest.json` and that the two rules and their code match `models/v3/manifest.json`, so
the file it writes is the file we submitted. The default is the v2 models with the v3 rules;
`--post none` writes the file of our second submission (v2 models alone, md5
`ddf10d9c44833549219a1b882f7929aa`). The first submission's models stay in `models/v1/` for
reference; v2 changed feature code that the v1 fingerprints guard, so the v1 file (md5
`075307b905359ae2a1bacbcdfc32d953`) is reproduced by its own repository,
https://github.com/Artsyadi/Bar-a-Herizon (tag `v1-submission`).

## Repository layout

| Path | What |
|---|---|
| `scripts/make_submission.py` | data bundle → submission file (the one command above) |
| `bh/` | the system: loading, features, models, segmenter, scoring helpers, BAR-ATH reader |
| `atd_cleaner/` | tracking cleaner (duplicates, jumps, speeds from positions, ball status) |
| `config/` | every setting: data split, features, model settings, segmenter costs, cleaner thresholds |
| `models/v2/` | the frozen models of this submission and their manifest (md5 of every file the output depends on) |
| `models/v3/` | manifest of the two physics rules applied on top of the v2 models (`config/v3_post.json`, `bh/v3post.py`) |
| `models/v1/` | the models of our first submission, kept for reference |
| `docs/` | technical document and the full feature list (`features_list.csv`) |
| `scripts/step*.py` | the training and evaluation pipeline on external data (below) |
| `tests/` | unit tests on synthetic data |

## Retraining from external data (optional)

The submission only needs the frozen models. To rebuild them, the external data goes in
`external/` (ignored by git):

- **IDSSE**, 7 matches (public, CC BY 4.0), processed into standardized tracking, five ball-degraded
  copies per match and H/A/D labels by our team's preprocessing repository
  `external-football-training-handoff-v1` (available to the organisers on request), placed at
  `external/external-football-training-handoff-v1/`;
- **Metrica Sports sample data**, 3 matches, used for comparison only.

Steps, in order (each is resumable; repeat a step until it reports "all done"). Set
`BH_MODEL_VERSION=v2` for the current feature set (the fast possession evidence):

```bash
export BH_MODEL_VERSION=v2                   # Windows PowerShell: $env:BH_MODEL_VERSION="v2"
python scripts/step12_clean_all.py           # clean every external unit
python scripts/step24_build_tables.py        # feature tables (every 5th frame)
python scripts/step41_d_model.py             # dead-ball model, 6-fold leave-one-match-out
python scripts/step42_ha_model.py            # possession model, same folds
python scripts/step52_grid.py                # segmenter switching costs on the dev folds
python scripts/step71_freeze.py --train d    # final models on all 6 dev matches
python scripts/step71_freeze.py --train ha
```

Checks and evaluation: `step22_check_d.py`, `step23_check_ha.py` (features), `step31_check_scorer.py`
(scoring), `step32_baseline.py` (organisers' baseline on external matches), `step72_test.py` (the
held-out test match), `step61_errors.py`, `step62_change.py`, `step62_weighted.py` (error review and
the two fixes we tested and did not keep). Tests: `python tests/test_segmenter.py`,
`tests/test_features.py` and `tests/test_cleaner.py` run on synthetic data; `tests/test_scoring.py`
also needs the organisers' repository in `more-than-a-hack-2027/`; `python tests/test_v3post.py`
tests the two rules.

The two rules need no training. `scripts/v3_post_check.py` re-scores them on the development matches
from the out-of-fold predictions of the steps above (v2 0.6135 → v3 0.6332 spellF1),
`scripts/v3_metrica_guard.py` is the one Metrica check, `scripts/v3_freeze.py` writes
`models/v3/manifest.json`, and `scripts/v3_test.py` scores the frozen v3 on the held-out test match
J03WPY (v2 0.653 → v3 0.671 spellF1, all five copies better).

## Data and acknowledgements

- **IDSSE:** Bassek, M., Rein, R., Weber, H., Memmert, D. An integrated dataset of spatiotemporal and
  event data in elite soccer. *Scientific Data* 12, 195 (2025).
  https://doi.org/10.1038/s41597-025-04505-y (CC BY 4.0). Data owner: Deutsche Fußball Liga (DFL);
  collected by Sportec Solutions.
- **Metrica Sports sample data:** https://github.com/metrica-sports/sample-data
- **BAR-ATH match data:** provided by the organisers of More than a Hack 2027 (Barça Innovation Hub,
  Metrica Sports, Mobile World Capital Barcelona), licensed for the competition only and not
  included in this repository.
- **Evaluation:** the organisers' `possession_metrics.py`, `validate_submission.py` and reference
  baseline from their repository.
