# Barça Herizon · Technical document

More than a Hack 2027, Phase 1 · possession labelling of BAR-ATH from tracking data\
Code: https://github.com/Artsyadi/Bar-a-Herizon-v3

## 1. Summary

We label every frame of the tactical video as home possession (`H`), away possession (`A`) or dead
ball (`D`) from the provided tracking data alone. The system has five parts:

1. a code-only cleaner for the automatic tracking (players and ball);
2. a **dead-ball model** that sees only the players;
3. a **possession model** that sees players and ball;
4. a **two-layer Viterbi segmenter** that turns frame probabilities into possessions;
5. two **physics rules** around the segmenter: a ball seen beyond a touchline or goal line is dead,
   and an opponent's touch inside a possession opens a short possession for that team.

Both models are gradient-boosted trees (LightGBM) trained on six external matches (IDSSE) whose
ball tracking we degraded to look like the provided data; a seventh IDSSE match is held out as the
final test. Nothing was trained, tuned or calibrated on the provided game, its video or its ground
truth.

On the held-out match, scored after the system was frozen and chosen, this submission reaches
**spellF1 0.671** (macroF1 0.909), against 0.219 for the organisers' baseline.

### The three versions, and the one we submit

We built the system in three versions. Each was frozen and chosen on our development matches
before it was scored on the held-out match, and each later version contains everything in the
earlier one.

| Version | What it is | Development, 6 matches × 5 copies | Held-out J03WPY, 5 copies | Metrica, 3 matches |
|---|---|---|---|---|
| **v1** | Cleaner, players-only dead-ball model, possession model, two-layer segmenter | 0.553 (macroF1 0.875) | 0.577 (macroF1 0.887) | 0.684 |
| **v2** | v1 plus fast possession evidence in the possession model (touches, receptions, players within 1 m; section 5), retrained | 0.614 (macroF1 0.886) | 0.653 (macroF1 0.902) | 0.720 |
| **v3** | The v2 models unchanged, plus two physics rules around the segmenter (section 5) | **0.633** (macroF1 0.894) | **0.671** (macroF1 0.909) | **0.737** |

All values are spellF1. **We submit v3**, for three reasons:

1. **It is the best version on every dataset we have.** Each step improved all six development
   matches (v1 → v2 by +0.061, v2 → v3 by +0.020). On the held-out match, v3 improved all five
   degraded copies over v2 (+0.017, close to its development gain). On the three Metrica matches,
   from another provider (the company that produced the provided tracking), it improved all three.
2. **It adds no learned parameters.** v3 uses the v2 models as they are. Its two rules are fixed from
   physics (the ball is out once it is over the line; a touch by the other team starts a
   possession), so there is nothing new that could overfit the training matches.
3. **It targets our two largest measured error sources**: late dead-ball starts and missed short
   possessions (section 9).

The v3 rules were the best of several post-processing ideas tried on the same development matches,
so their development gain is likely somewhat optimistic; the held-out and Metrica results were
measured after the choice and point the same way. The risks specific to the provided data are in
section 10. All three files are reproducible (section 11).

## 2. Approach, and why

**Tracking data, not video.** We work from the tactical ATD. It is one record per submission frame
(video frame = ATD `frame_id` − 1), already in pitch coordinates, and gives every player's position
directly. The broadcast recording runs on its own timeline (cuts, replays, 164,488 frames) with no
mapping to the tactical frames the submission is indexed to. Learning possession from video would
also need labelled video, and training on the provided video is not allowed. The tracking is
noisy, but noise in coordinates can be handled with code, which is where we put the effort.

**Two questions, two models.** "Is the ball in play?" and "which team has it?" depend on different
evidence, so we answer them separately:

- **Dead ball is decided from the players only.** When play stops, the players show it: both teams
  slow down together, shapes change (walls, kick-off formations, players gathering in the box or at
  a touchline). Using players only means the dead-ball decision does not depend on the ball, which
  is missing on about 41% of the graded frames and which, in our external data, freezes in place
  during dead play. A ball-based dead-ball rule would learn that freeze rather than football.
- **Possession is decided on live frames only**, from the ball and the players around it, plus
  team context that still works when the ball is missing (team compactness, movement towards goal,
  how high the team stands).

**Possessions, not frames.** The ranking metric matches whole possessions, so frame-by-frame
classification is not enough: it flickers and creates more than twice as many possessions as
exist. A segmenter that charges a cost for every switch produces clean possessions and was the
single largest improvement (spellF1 0.332 → 0.614 on our development matches).

## 3. Assets: used and dropped

| Asset | Use |
|---|---|
| Tactical ATD (`_atd.xml` + `_atd.txt`) | **Used**: the only input of the system |
| Tactical video | Watched to understand the match and possession; not an input and never used for training or for checking output |
| Broadcast video and its ATD | **Dropped**: separate timeline, no mapping to the submission frames |
| Homography | **Dropped**: the ATD is already in pitch coordinates, and the camera information it adds has no counterpart in our external training data |
| Smart Tagging | **Dropped**: we concentrated on the tracking; not used in this submission |
| Public 5-minute ground truth | Pipeline check only: frame alignment, team mapping (first metadata team = `H`), label meaning. Never used for training, tuning or any choice |
| Organisers' scorer, validator and baseline | Used to score external matches, validate the file and compare |

## 4. Training data

All labelled data comes from other games. No label of any kind on the provided game was used.

| Dataset | Matches | Labels and who produced them | Amount | Use |
|---|---|---|---|---|
| **IDSSE** (Bassek et al., 2025; CC BY 4.0) | 7: Bundesliga 2022/23 (J03WMX, J03WN1) and 2. Bundesliga 2022/23 (J03WOH, J03WOY, J03WPY, J03WQQ, J03WR9) | Per-frame ball status (alive/dead) and possessing team from the official tracking data of the data provider (Sportec Solutions for the DFL, TRACAB optical tracking). We converted them mechanically: `D` where the ball is dead, otherwise the possessing team. No person on our team labelled anything | 1,002,644 graded frames (about 11 hours), 3,061 possessions | 6 matches for training and development, 1 (J03WPY) held out as the final test |
| **Metrica Sports sample data** | 3 anonymised matches | Derived from Metrica's published event data | 429,923 graded frames, 1,038 possessions | Comparison only; never trained on |

**Making the external data look like the provided data.** The IDSSE ball is almost always visible;
in the provided tracking it is missing on about 41% of graded frames (DATA.md). We therefore made
five degraded copies of every IDSSE match: ball frames removed in gaps, with ball availability
drawn around 59% (between 50% and 70%) and gap lengths taken from the real observation gaps of the
Metrica raw tracking. The players were left unchanged. Every model and choice uses all five copies
and reports the mean and the worst copy.

## 5. The pipeline

**Cleaning** (all thresholds from physics or external data, none from the provided game):

- duplicate tracks merged, impossible jumps and speeds removed (players above 12.5 m/s, ball above
  its physical limit);
- player speed recomputed from positions (centred difference over ±2 frames), because the provided
  speed channel's declared unit does not match its values and the external data has no speed;
- the ball gets a status per frame: observed, estimated across a short gap, or unavailable; nothing
  is invented across long gaps.

**Features** (393 columns, listed one by one in `docs/features_list.csv`):

- dead-ball model, 66 player-only values (264 columns with time windows): share of slow and running
  players per team, both teams slowing together, deceleration, team width, depth, compactness and
  position, defensive line height, kick-off shape, tight groups (walls), players near a touchline
  or in a penalty area, and how players move relative to their nearest opponent;
- possession model, 51 values (129 columns): ball state and motion, nearest player per team and
  the distance margin between the teams, players within 2 and 5 m of the ball, ball–carrier
  agreement, which team has been nearer over the last 1 and 3 s, the last reliable margin before a
  ball gap, and team context; plus 14 values of fast possession evidence: players within 1.0 and
  1.5 m of the ball per team, whether only one team is that close, the ball's speed relative to the
  nearest player, seconds since the last touch and the last reception by each team, the team of the
  last touch, and the share of the last 0.5 s in which the home team was nearer. A touch is a
  change of ball velocity of at least 3 m/s between 0.4 s before and 0.4 s after with a player
  within 1.5 m (the nearer team touches); a reception is the ball slowing by at least 2.5 m/s next
  to a player. Both need the ball observed at all three instants;
- every value also as 1 s and 5 s averages and a 2 s change; windows never cross half-time;
- missing inputs stay missing (never zero); a feature is left blank when fewer than 7 players of a
  team are tracked.

**Models.** LightGBM, 400 rounds, learning rate 0.05, 31 leaves, minimum leaf 200 (dead ball) / 500
(possession), fixed before any result and not tuned. Trained on every 5th frame. The dead-ball
model uses one copy per match (its inputs do not depend on the ball); the possession model uses all
five.

**Segmenter.** Two layers, each a two-state Viterbi with a cost per switch, in log-odds units:

1. dead vs live from the dead-ball probabilities (cost 64);
2. inside each live interval, home vs away from the possession probabilities (cost 8), decided
   afresh in every interval.

Both costs were chosen on the development matches from a 5 × 5 grid fixed before running.

**Two physics rules (v3).** Both are fixed from physics, nothing in them is learned, and the
models and segmenter costs are the v2 ones (`config/v3_post.json`, `bh/v3post.py`):

1. **Ball out of play.** Before the dead-ball layer, every frame whose ball is *observed* (not
   estimated) more than 0.5 m beyond a touchline or goal line gets dead-ball probability 1: the ball
   is out once it is wholly over the line, and 0.5 m covers the ball's radius (0.11 m) and tracking
   error. Lines come from the pitch size (105.3 × 68 m for BAR-ATH). It gives the players-only
   dead-ball model the one ball event that is certain. On development data 99.3% of these frames
   are dead in the ground truth, and the rule touches about a fifth of all dead-ball starts (the
   others are whistles, or the ball is not seen at the line).
2. **Touch candidates.** After the segmenter, inside a predicted possession of team X, a touch by
   the other team Y (the touch detector of the possession features, unchanged: a velocity change of
   at least 3 m/s with a Y player within 1.5 m) relabels the frames from that touch to the next
   touch by X as Y, for at most 2 s, stopping at a dead ball or half-time; it is applied only if it
   lasts at least 3 frames. This recovers short possessions the segmenter absorbs; 82% of these
   candidates are real possessions on development data.

## 6. The possession rules we implemented

We wrote our own possession rulebook before building, and the system follows it:

- **Dead ball** is a stoppage, never a possession, and comes from the players only. Frames outside
  the two halves are `D`. The file never contains `X`.
- **Restarts.** After every dead spell, possession is decided afresh from what happens after the
  restart. Nothing carries across a stoppage, so `H` → `D` → `H` is always two possessions. We do
  not model who "owns" a dead ball (the ground truth does not ask for it).
- **Possession changes on effective contact.** Our first rulebook changed possession on control
  only. The challenge states that the ground truth prevails over the written definition, and the
  reference possession flags switch at the opponent's first touch: an opponent touch that changes the ball's speed or direction transfers possession, even
  without control. We rewrote the rule accordingly. The labels we learn from already follow it, so
  nothing was relabelled. The segmenter alone still absorbs many brief contacts, because a switch
  must be paid for by sustained evidence; the touch-candidate rule (section 5) reinstates a short
  possession where a touch by the other team is detected.
- **Loose and contested balls stay with the last team that made effective contact** until the
  other team's evidence or touch is strong enough to switch.
- **No minimum duration.** A short possession survives when its evidence is strong enough; there is
  no fixed length anywhere.
- **Ball not visible.** Ball features are left blank, never guessed; the model then relies on team
  context and on the last reliable ball–player relationship before the gap.

## 7. How we validated

- **Development:** 6 IDSSE matches, 6-fold leave-one-match-out: every match is predicted by models
  that never saw it. All choices (segmenter costs, keep or drop a change) were made on these folds.
- **Final test:** J03WPY, chosen before any result as the most typical IDSSE match by label
  statistics, scored with the frozen system once per version, all five degraded copies, mean and
  worst: 0.577 for v1, 0.653 for v2, 0.671 for v3. It was never used to train or to choose
  anything; for v2 it served as a no-regression check against a threshold written down before the
  look, and nothing was changed afterwards.
- **Comparison:** the 3 Metrica matches, another provider never used for training or choices.
- **The two physics rules (v3)** were judged on the 6 development matches × 5 copies with the same
  keep rule as every other change (more than +0.016 spellF1 and at least 5 of 6 matches better).
  Since J03WPY had already been looked at for v2, we did not use it for this decision. Two checks
  were written down before looking instead: the rules must not lose ground on the player-degraded
  development copy (player positions degraded as well, at the published noise level), and one
  single look at Metrica must stay at or above 0.700 spellF1. Both held. Only after v3 had been
  chosen and frozen did we score it on J03WPY (`scripts/v3_test.py`): 0.671 against v2's 0.653,
  all five copies better. That result was reported as it came and changed nothing. The rules were
  the best of several post-processing ideas tried on the same development matches, so their
  development gain (+0.020) is likely somewhat optimistic; the held-out gain (+0.017) is close to it.
- **Baseline:** the organisers' reference (nearest player + ±50-frame majority vote), reproduced
  exactly (identical output to their code) and run on the same matches.
- **Freeze:** models, configuration and code fingerprinted (md5) before the test; retraining gives
  byte-identical models.
- **Leakage test:** feeding the dead-ball model a shuffled ball, or another copy's ball, leaves its
  predictions identical.

## 8. Results

spellF1 decides the ranking; macroF1 is the tie-break. On one full match, spellF1 differences below
about 0.04 are noise.

| Data | v3 spellF1 | v2 spellF1 | v2 macroF1 | Baseline spellF1 | Baseline macroF1 |
|---|---|---|---|---|---|
| Development, 6 matches × 5 copies (out of fold) | **0.633** (macroF1 0.894; all 6 matches better) | 0.614 | 0.886 | 0.216 | 0.457 |
| Development, players degraded as well (out of fold) | **0.602** (all 6 matches better) | 0.587 | | | |
| Final test J03WPY, 5 copies (frozen system) | **0.671** (worst copy 0.641; macroF1 0.909; all 5 copies better) | 0.653 (worst copy 0.619) | 0.902 | 0.219 | 0.458 |
| Metrica, 3 matches (never used; one look for v3) | **0.737** (all 3 matches better) | 0.720 | 0.883 | 0.427 | 0.806 |

Our first submission scored 0.553 / 0.577 / 0.684 on the development, test and Metrica rows; the
fast possession evidence is the only difference between v1 and v2. Between v2 and v3 the models are
identical; the difference is the two rules of section 5. With them the system predicts 355
possessions per development match instead of 346 (440 in the ground truth), and 347 instead of
333 on the test match (421 in the ground truth). On the test match the dead/live decision reaches
F1 0.966 with v3 (0.960 with v2; live as positive class) and, on live frames, the right team is
predicted 89% of the time. Each version improved all six development matches, and Metrica, another
provider, improved as well.

**What the models rely on.** The dead-ball model leans mostly on the share of slow players and on
both teams slowing together (about 70% of its decisions), then team shape. The possession model
leans on the team of the last touch and which team has been nearer the ball over the last 3
seconds (together about a third of its decisions) and on team context: the team in possession
spreads out and moves forward (about 40%). The fast evidence added in v2 carries about a third of the possession model's total split gain.

## 9. What we tried and did not keep

We kept a change only if it raised mean development spellF1 by more than 0.016 (the 0.04 noise
band over six matches) and improved at least 5 of the 6 matches; the rule was fixed before testing.

| Change | Result | Decision |
|---|---|---|
| **Fast possession evidence** (touches, receptions, players within 1 m, see section 5) | +0.061, all 6 matches better | **kept** |
| A change-of-possession detector making switches cheaper where a change looks likely | +0.009, 5 of 6 matches (combined with the weighting below: +0.006, 3 of 6 on top of the fast evidence) | dropped |
| Extra training weight on frames within 1 s of a possession change | +0.012, 5 of 6 matches | dropped |
| Different switching costs for dead→live and live→dead | +0.000 | dropped |
| Degrading the players as well as the ball in training | +0.009, 5 of 6 matches | dropped |
| Training on every frame instead of every 5th | −0.000 | dropped |
| Additional dead-ball and possession features | +0.011 at best, 5 of 6 matches | dropped |
| Labels derived from event data for IDSSE | replaced by the provider's per-frame ball status and possession flags, which are frame-exact | dropped |
| A ball-based dead-ball rule | the ball freezes during dead play in the external data, so such a rule learns a data artefact | dropped in favour of players only |

For v3 we first measured where the remaining errors are, with every other part made perfect in turn
on development data: perfect dead-ball boundaries would add +0.078 (almost all of it the start of
dead spells, predicted about 0.4 s late at the median), perfect possessions under 2 s +0.133. Then,
on the v2 outputs, without retraining:

| Change (v3) | Result | Decision |
|---|---|---|
| **Ball out of play forces dead ball** (section 5) | +0.011, all 6 matches better | **kept**, with the next row |
| **Touch candidates** (section 5) | +0.008, all 6 matches better | **kept** |
| Both together | +0.020, all 6 matches better | **this submission** |
| Shifting every possession change earlier or later by up to 0.8 s | every shift worse: the changes are on time (median delay 0 frames); the loss is changes not detected at all | dropped |
| Switching costs that depend on the ball (dearer when unseen, cheaper near a touch) | +0.004, all 6 matches better | dropped |
| Many cheap candidate possessions filtered by a small learned verifier | +0.004; a perfect verifier would reach only +0.010 | dropped |
| A looser touch detector (2 m, 2 m/s) | +0.003, 4 of 6 matches: more false touches than recovered possessions | dropped |

## 10. Limitations: where it fails

- **Short possessions.** The system predicts about 21% fewer possessions than there are (346 vs
  440 per match on development data). Most possessions shorter than 2 seconds (30% of all) are
  missed: inside one that lasts under a second, the possession model favours the right team on
  only 47% of frames (64% for 1–2 s). The segmenter is not the main cause: making switches
  cheaper finds only a few more of them and adds many false ones. On development data, catching
  these perfectly would be worth about +0.16 spellF1.
- **Dead-ball boundaries.** Short stoppages are sometimes missed and dead spells start and end
  late; perfect dead/live boundaries would be worth about +0.08 on development data. The ball-out
  rule fixes the starts it can see, about a fifth of them.
- **Line positions in the provided tracking.** The ball-out rule trusts the touchlines and goal lines
  of the provided coordinates (pitch 105.3 × 68 m). On development data its gain shrinks from +0.020
  to +0.013 when the margin grows from 0.5 m to 1 m; a line offset of 0.5 m or more in the provided
  tracking would shrink it, and a large offset could mark balls still in play as dead. We did not
  check this on the provided game, because that would mean reading its values.
- **Short possessions the ball cannot show.** Of the short possessions still missed, 45% have no
  observed ball at all and most of the rest show no detectable touch, so no rule on the ball can
  recover them.
- **Noisier input than in training.** Our degradation removed ball frames only. The provided
  tracking also has player gaps, fragmented tracks and identity switches, which the models did not
  see in training. Half of the possession model's evidence is team context that needs at least 7
  tracked players per team.
- **Ball noise.** The fast evidence relies on the ball's velocity changing at a touch. In a stress
  test on external data with independent per-frame ball jitter, the system stays ahead of our
  first submission up to 0.5 m of jitter and falls behind it at 1.0 m. The real noise of the
  provided ball is unknown to us; Metrica's sample data, produced by the same company that
  produced the provided tracking, was our closest check and improved.
- **Timing convention.** In our analysis of the external data the possession flag follows touches, not proximity:
  it switches at the new team's first touch, typically a few tenths of a second after the nearer
  team changes (median 0.4 s, most within 2 s). If the provided ground truth times its switches differently, our
  changes will be systematically early or late, and spellF1 is sensitive to timing.

With more time we would give the dead-ball model the ball itself (inside the window the players
define), learn short possessions with a model that reads the raw movement of the ball and the
nearest players over a few seconds, and degrade the players as well as the ball in training. We
would also try the camera information in the homography where the ball is missing, which first
needs training data with a camera (or a realistic simulated one).

## 11. Reproduction

```bash
pip install -r requirements.txt                       # Python 3.10, pinned versions
python scripts/make_submission.py                     # data bundle -> outputs/Barça Herizon.csv (v2 models + v3 rules)
```

The data bundle goes in `more-than-a-hack-2027/data/` (or `MTAH_DATA_DIR`); see the README. The
script checks the frozen models and configuration against their fingerprints, reads and cleans the
tactical ATD, builds the features, applies the two models, the segmenter and the two rules, writes
the file and runs the organisers' validator. Retraining from the external data is described in the
README. The earlier versions are reproducible too:

| Version | Command | md5 of the file |
|---|---|---|
| **v3 (this submission)** | `python scripts/make_submission.py` | `bc3a2a71ffae604b98e798ba843dbd67` |
| v2 | `python scripts/make_submission.py --post none` | `ddf10d9c44833549219a1b882f7929aa` |
| v1 | `python scripts/make_submission.py` in https://github.com/Artsyadi/Bar-a-Herizon (tag `v1-submission`) | `075307b905359ae2a1bacbcdfc32d953` |

## 12. Compliance with the rules

- Labels are produced by an algorithm; no hand labels or hand corrections on the provided game.
- No use of the private ground truth; the public 5 minutes served only to check the pipeline.
- No training, fitting or calibration on the provided game or its video; all training data is from
  other games.
- No human-produced data about the provided game from any source.
- The repository contains no match data and regenerates the submitted file exactly.

## 13. Data and acknowledgements

- Bassek, M., Rein, R., Weber, H., Memmert, D. An integrated dataset of spatiotemporal and event
  data in elite soccer. *Scientific Data* 12, 195 (2025). https://doi.org/10.1038/s41597-025-04505-y
  (CC BY 4.0). Data owner: Deutsche Fußball Liga (DFL).
- Metrica Sports sample data, https://github.com/metrica-sports/sample-data
- BAR-ATH match data and evaluation tools: More than a Hack 2027 organisers (Barça Innovation Hub,
  Metrica Sports, Mobile World Capital Barcelona). The match data is licensed for the competition
  only and is not included in our repository.
