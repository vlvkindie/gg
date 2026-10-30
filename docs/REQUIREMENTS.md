# Requirements and evidence

This checklist follows every task, rule, measurement, check and submission item
in the supplied starter README. Repository setup is complete independently of
benchmark admission. Pending items remain explicit.

## Repository and reproducibility

- Starter in the user-selected `vlvkindie/gg` repository: prepared for `main`.
  The user must confirm this repository is registered with the hackathon platform.
- Standard root layout: `solution/`, `harness/`, `tests/`, `task/`, `templates/`,
  README, LICENSE and NOTICE. Supporting scripts and docs are included.
- Protected starter files: byte-identical hash guard, locally passing.
- Original kernels and patches: contained in `solution/`; example attributed.
- Pinned training dependencies: `requirements-train.txt`, matching the template.
- Kaggle notebook: `kaggle_gg.ipynb`, configured for `gg` and `main`.
- CPU reproduction and CI: `requirements-cpu.txt`, runner and GitHub Actions.
  Local CPU checks passed. Remote CI status must be checked after publication.
- Model, public rows, batches, steps, LoRA including dropout, optimizer, learning
  rate schedule, Soup switches and allocator: unchanged by this solution.
- Sources: `SOURCES.md`; kernel docstrings include mathematical derivations.
- Jury access: repository is currently private. Public visibility or jury access
  through the platform remains an account/submission requirement.

## All ten official checks

1. Every base and solution run exits zero: pending full T4 run.
2. No banned libraries or runtime package changes: static imports pass; runtime
   loaded-module and package checks remain pending T4.
3. Original Triton kernels launch in every training run: three kernel functions
   recognized statically; real training launches remain pending T4.
4. Original kernels take at least 5% of timed training: unmeasured. The example's
   time never counts toward this threshold.
5. Same task and Soup outputs: settings and computation paths retained; exact
   adapter, tokens and row identity, plus entropy/accuracy/row-loss tolerances,
   require full base-versus-solution checks.
6. Loss curve: first-step relative gap at most 1%, mean relative gap at most 3%.
   Pending T4 training.
7. Held-out adapter loss: at most 0.5% above the base. Pending T4 evaluation.
8. Time and memory: no more than 10% above the base. Pending measurement.
9. Clean T4 measurement: harness uses GPU 0, unchanged allocator, idle-start and
   peak checks. Only the official run can establish this condition.
10. Exactly 79 Soup regressions: passed locally with `SOUP_APPLY_SOLUTION=1`.
    Local Triton is unavailable, so actual GPU-patched regression coverage remains
    part of T4 preflight. CPU pass alone does not establish all-ten-check admission.

## Scoring and reporting

- Speed and memory gains: no values claimed. Ratios come from the real harness.
- Improvements below 5% score as a tie; README does not claim unmeasured gains.
- README covers what changed, how to reproduce, results and limitations.
- Final reported values must match `runs/result.json` and `runs/checks.json`.
- Both training-time kernel share and adapter quality are release gates.

## Submission materials

- Root `SUBMISSION.md`: filled with repository/material links and explicit status.
- Final admitted commit: pending a passing full T4 run.
- Public Kaggle saved-version URL: pending account execution/publication. Its
  `COMMIT` must equal the final validated SHA; insert the same URL in README and
  SUBMISSION.
- Presentation: repository materials cover operations, kernel equations, results
  status, limits and next steps. Measured comparisons await T4 results.
- Demo: narration/recording script covers a video under three minutes. A final
  recording must show actual measurements; none are fabricated.
- Team details: user input pending, at most four members and one team result.
- Platform submission and jury access: user/account step, not established by a
  Git push. No submission or public publication has been claimed.
