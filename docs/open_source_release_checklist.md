# IndusTSFM open-source release checklist

This checklist is for the maintainer's first public push. It is intentionally
separate from the experiment protocols: it checks repository hygiene and
distribution boundaries, not scientific validity alone.

## Before the first push

- [ ] Replace the generic contributor copyright line with the actual copyright
      holder if required by the maintainer.
- [ ] Confirm whether Apache-2.0 is the desired code license; update
      `LICENSE`, `NOTICE`, `CITATION.cff`, and `pyproject.toml` together if it
      changes.
- [ ] Set the public repository URL in the README and `CITATION.cff` after the
      GitHub repository is created.
- [ ] Run `git status --short --ignored` and verify raw data, checkpoints,
      caches, results, logs, `.env` files, and temporary directories are not
      staged.
- [ ] Run a secret scan on the exact files intended for the first commit.
- [ ] Run `python -m pip install -e ".[dev]"`, `python -m pytest -q`,
      `python -m ruff check src scripts tests`, and `python -m compileall -q
      src scripts tests` in a clean environment.
- [ ] Review `THIRD_PARTY_NOTICES.md` against the exact dependency and model
      versions used for the release.

## Before each tagged release

- [ ] Re-run the relevant artifact audits from a clean raw-data directory.
- [ ] Do not promote synthetic smoke results to benchmark evidence.
- [ ] Include resolved configs, source manifests, split/scaler manifests,
      metrics, predictions, and audit output for any published result bundle;
      exclude raw source files and downloaded weights unless redistribution is
      explicitly permitted.
- [ ] Record known limitations, dependency constraints, and model/data terms
      in the changelog and release notes.
- [ ] Enable branch protection, private vulnerability reporting, and CI on the
      public repository.
