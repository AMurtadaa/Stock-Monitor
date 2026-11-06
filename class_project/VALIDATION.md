# Implementation verification

The class branch was exercised in an isolated Windows Python 3.12 environment.

Verified:

- Actual public PhraseBank archive download and parsing, not just fixture input.
- 3,453 input rows -> 3,448 unique clean examples -> 2,413/517/518 split.
- Actual DistilBERT download, two-step CPU smoke training, save and reload check.
- Smoke evaluation on 12 development examples, with VADER comparison and plots.
- Offline tests for duplicates/conflicting labels, immutable split identities,
  named checkpoint labels, batch consistency, empty/long text, news deduplication,
  20-per-day cap, 48-hour availability delay, future-news exclusion, actual label-end
  purging, output isolation, and known metric values.
- Synthetic end-to-end financial example construction using upstream features and
  actual LightGBM training; both variants use identical final rows and different
  feature lists. Synthetic scores are not financial research evidence.
- Notebook JSON/code-cell compilation and class-project Ruff checks.

Still requires execution with the user's data/Colab:

- Three full sentiment runs and held-out benchmark evaluation.
- Historical news coverage audit, real financial snapshots and matched evaluation.
- Full-checkpoint demo and interpretation review.
- Final report/slides populated from actual results.

Commands:

```text
python -m class_project check
python -m pytest class_project/tests -q
ruff check class_project
python -m class_project prepare
python -m class_project train --smoke
python -m class_project evaluate --smoke
```

Source-level tests do not establish sentiment accuracy or financial improvement.
The main application and its scheduler were not run or modified by these checks.

## Clean notebook V2 repair

The user's saved original notebook is preserved. Its cells contained pip repair
commands before environment creation, an early `run("check")` before its helper,
and a later non-streaming helper that replaced output streaming. V2 removes these
competing paths and imports a single standard-library command runner.

Local Windows CPU verification includes 17 passing tests: streamed stdout/stderr,
heartbeat during silence, nonzero-exit handling, setup ordering (including pip
bootstrap), failed setup stopping, smoke artifacts required before full training,
changed checkpoint rejection, notebook code compilation, and the existing data/
financial/backend correctness checks. Class-project Ruff passes excluding the
preserved original notebook, whose out-of-order cell is intentionally not edited.

The new runner was also exercised against actual CLI dependency checks, frozen
dataset preparation, and the existing real DistilBERT smoke checkpoint/evaluation.
Reuse is explicitly printed. Checkpoint and dataset checksums are verified.
This was not a new three-seed training run and not a Colab GPU execution.
Drive authorization, current Colab installation and full training still require
running V2 in the user's GPU runtime. No final experimental metrics are invented.
