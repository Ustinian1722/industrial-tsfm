# Data policy and local reproduction

Raw datasets are intentionally not versioned in this repository. The `data/`
directory contains only this policy document and an empty `data/raw/.gitkeep`
placeholder. Downloaded files are ignored by Git and remain the user's
responsibility under the upstream dataset terms.

## C-MAPSS

Put the following files in `data/raw/` for the FD001 baseline:

```text
train_FD001.txt
test_FD001.txt
RUL_FD001.txt
```

The original NASA Dashlink description identifies four C-MAPSS simulation sets, run-to-failure engine trajectories, and several recorded sensor channels. NASA's current data catalog notes that C-MAPSS downloads are temporarily unavailable. For local reproduction, `scripts/fetch_cmapss_fd001.py` can retrieve an unmodified public mirror after an explicit `--accept-mirror` flag. The run manifest records the mirror URL and file hashes; the data are never committed by this repository.

The mirror is a transport mechanism, not a replacement for NASA provenance. If the official NASA files are obtained later, place those files at the same paths and rerun the integrity checks.

## Other registered domains

The fetchers and configs for the other domains are:

| Domain | Fetch command | Local directory | Protocol |
|---|---|---|---|
| UCI Gas Turbine | `python scripts/fetch_gas_turbine.py --accept-mirror` | `data/raw/gas_turbine/` | `docs/cross_domain_protocol.md` |
| Tennessee Eastman | `python scripts/fetch_tennessee_eastman.py --accept-source` | `data/raw/tennessee_eastman/` | `docs/tennessee_eastman_protocol.md` |
| NASA/IMS Bearings | `python scripts/fetch_ims_bearings.py --accept-source` | `data/raw/ims_bearings/` | `docs/ims_bearings_protocol.md` |

The IMS archive is large. Use `--metadata-only` to write a provenance contract
without downloading it. No fetch command is executed by the test suite or CI.

## Rules for contributions

- Do not commit raw data, source archives, derived private data, or credentials.
- Preserve `source_manifest.json` locally for every reproducible experiment.
- Do not treat a public transport mirror as a new license or new provenance.
- Check the relevant entries in `THIRD_PARTY_NOTICES.md` before redistributing
  any data, model output, or release bundle.
