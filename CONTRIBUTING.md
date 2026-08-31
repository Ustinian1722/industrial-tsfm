# Contributing to Industrial TSFM

Thank you for contributing. The project values reproducibility, explicit
benchmark boundaries, and small changes that can be audited.

## Development setup

Python 3.10 or newer is supported. From the repository root:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Optional model integrations are intentionally separate because they download
large checkpoints and can impose incompatible dependency constraints:

```powershell
python -m pip install -e ".[tsfm]"
python -m pip install -e ".[moirai]"
python -m pip install -e ".[peft]"
```

Run the same checks used by continuous integration:

```powershell
python -m pytest -q
python -m ruff check src scripts tests
python -m compileall -q src scripts tests
```

## Data and experiments

- Never commit raw datasets, model checkpoints, Hugging Face caches, generated
  results, credentials, or private logs.
- Use the fetch scripts and their explicit source-acceptance flags. Preserve
  `source_manifest.json` locally so results remain reproducible.
- Add synthetic fixtures to `tests/` for loader and contract changes. Tests
  must not download external data or model weights.
- Do not select models or tune hyperparameters on held-out target labels.
  Preserve entity/time isolation and source-only preprocessing.
- New reported results need a resolved config, split/scaler manifest,
  predictions, metrics, resource metadata, and a passing audit script.

## Code and documentation style

Keep the common dataset/model interfaces stable. Prefer typed, focused
functions and explicit configuration fields over hidden defaults. Update the
relevant protocol document and both READMEs when changing a benchmark
contract, data source, model input contract, or reproducibility rule.

## Pull requests

Please include:

1. a short problem statement and the proposed change;
2. tests or a clear reason tests are not applicable;
3. the exact validation commands and their outcomes;
4. any data, model, dependency, or license impact;
5. a note confirming that no raw data, weights, secrets, or private artifacts
   are included.

Do not include cherry-picked headline results without their complete artifact
and audit context. Contributions are accepted under the repository license;
by submitting a contribution, you confirm that you have the right to do so.
