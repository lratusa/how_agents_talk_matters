# Contributing

Thank you for your interest. This repository is a finished controlled study
plus a reproduction package; the most valuable contributions are
**replications and extensions**, not refactors.

## What we welcome

- **Replications** on other models, benchmarks, or agent counts (see the
  open-problems list in `README.md`). The cached initial-answer protocol in
  `run_experiment.py` is the part to keep: apply every pairing rule to the
  same response set, or the comparison is not controlled.
- **New routing or repair methods** evaluated on our archived dataset
  (zero API cost; see `docs/open_data_guide.md`).
- **Bug reports** with a reproducing command. Deterministic bugs (parsing,
  matching, statistics) are always in scope.
- **Analysis improvements**: new metrics on the existing audit trail
  (`results/full_analysis_*.json`).

## Ground rules

- **Never modify raw records.** Files under `results/` that begin with a
  benchmark name, and anything derived from `data/`, are append-only
  evidence. Fix bugs in the code, then re-derive.
- **Run the offline suite before sending a PR**: `python test_pipeline.py`
  (138 checks, mock model, no API key needed).
- **Do not commit secrets.** Keys come from environment variables or a
  local dotenv file pointed to by `FPRR_KEY_FILE` (see `src/config.py`).
- Match the existing code style: stdlib-first, no new dependencies without
  a justification in the PR description.

## How to cite work built on this repository

If your project uses this code, the archived dataset, the figures, or the
manuscript, cite it as shown in `CITATION.cff` / `README.md`. If you publish
a replication or extension, we would be glad to hear about it — open an
issue titled `replication: <short description>` and we will link it from
the README.

## Process

1. Open an issue describing the change before large work.
2. Fork, branch, PR against `main`. Keep diffs scoped.
3. For anything touching statistical conclusions, include the recomputed
   numbers in the PR.
