# Submission draft — T4 validation pending

Repository: https://github.com/vlvkindie/gg

Implementation location: the repository's `main` branch, with kernels in `solution/`.
Final validated commit: pending a passing T4 run.

## Links and materials

- [Repository](https://github.com/vlvkindie/gg)
- [Configured Kaggle notebook file](kaggle_gg.ipynb)
- Public saved Kaggle notebook URL: pending actual account execution/publication.
- [Presentation draft](docs/presentation.pptx), with results explicitly pending.
- [Demo recording script](docs/DEMO_SCRIPT.md), 2 minutes 40 seconds.
- Final demo video URL: pending recording of the actual T4 output.
- [All requirements and status](docs/REQUIREMENTS.md)
- [Current results and limitations](RESULTS.md)

## Team

Team name, participant names, roles, Kaggle handles and captain contact: awaiting
user-supplied details. Repository owner: `vlvkindie`. Do not infer that the owner
is the team's official name or that membership is already registered. The limit
is four participants and one submitted result per team.

## Kernels

Original kernels:
- `soup_rmsnorm_forward`, `soup_rmsnorm_backward` in `solution/rmsnorm.py`.
- `soup_rope_pairs` (forward and backward specializations) in `solution/rope.py`.

See `SOURCES.md` for equations and attribution. The supplied SwiGLU example is
retained and explicitly excluded from the original-kernel claim.

Time, peak memory, speedup, memory ratio, eval-loss delta, and training-time kernel
share: **not measured on T4**. All-ten-checks status: **not established**.

## Submission gates

- [x] README explains the operations, launch commands, local results and limits.
- [x] Starter layout, original-kernel code, attribution and presentation prepared.
- [x] CPU tests and protected-file guard pass locally.
- [ ] Full T4 run passes every official check, including original-kernel share.
- [ ] Copy actual base/solution time, memory, held-out loss and ratios here.
- [ ] Set notebook COMMIT to the validated SHA and save a public Kaggle version.
- [ ] Put the same public saved-version URL in README and this file.
- [ ] Record the results demo, finish the measured-results presentation and add links.
- [ ] Fill team details, arrange jury access and submit on the hackathon platform.

Run `python scripts/run_validation.py full` on T4, then use `scripts/report.py`
for actual measurements. This repository is implementation-ready; competition
admission remains unverified until the official checks and external steps finish.
