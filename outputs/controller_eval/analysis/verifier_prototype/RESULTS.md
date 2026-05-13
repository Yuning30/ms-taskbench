# 2-feature verifier prototype - results

Validates the analysis claim that target reach + min neighbor distance
capture almost all c9+c18 outcome variance.

Trained on in-sample c18 (500 scenes), tested on OOS c19 (500 scenes).

## Variant 1: binary (all 500 scenes)
Includes the OOW scenes (always-fail) and the always-success near-base scenes.

- ROC-AUC:    0.9223
- PR-AP:      0.9766
- Test F1:    0.9375
- Test acc:   0.9000
- Test prec:  0.9214
- Test rec:   0.9542

## Variant 2: reachable-only (tx_r <= 0.84m)
Drops 41 OOW scenes from in-sample and 41 from OOS. This is the
"verifier's real job": predict success on scenes the controller will
actually attempt.

- ROC-AUC:    0.8693
- PR-AP:      0.9761
- Test F1:    0.9389
- Test acc:   0.8908
- Test prec:  0.9035
- Test rec:   0.9771

## Variant 3: 2 features only (no source one-hot)
Same as variant 2 but using only (tx_r, min_nbr_d). Tests whether
scene source (random/canonical/templated) adds information.

- ROC-AUC:    0.8748
- PR-AP:      0.9769
- Test F1:    0.9434
- Test acc:   0.8974
- Test prec:  0.8950
- Test rec:   0.9975

## Interpretation

Even a 2-feature logistic regression captures most of the outcome
signal. The verifier doesn't need to be elaborate; the failure modes
the c9+c18 controller exposes are largely determined by where the
target sits in (reach, crowding) space. Figures 01-03 visualize ROC,
PR, and the chosen-threshold confusion matrix.
