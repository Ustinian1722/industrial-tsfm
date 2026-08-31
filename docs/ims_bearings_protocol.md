# NASA/IMS Bearings protocol

This is the mechanical-vibration extension contract. It is deliberately a
feature-level forecasting benchmark, not a claim that raw accelerometer values
are directly comparable to the process and turbine domains.

## Provenance and raw-data boundary

The registered source is the public NASA/IMS catalog distribution:

- catalog: <https://catalog.data.gov/dataset/ims-bearings>
- distribution: `https://data.nasa.gov/docs/legacy/IMS.zip`
- NASA repository entry: <https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/>

Run `python scripts/fetch_ims_bearings.py --accept-source` only when the local
machine has room for the large archive and its extracted files. The fetcher
records the archive hash, signal-file hashes, source links, and the local-only
raw-data note in `source_manifest.json`. `--metadata-only` records the source
contract without downloading the archive.

## Entity and feature contract

The loader reads one raw signal file as one condition cycle. Each bearing is an
entity, and the row index is the raw-file cycle. The default channel groups are:

- `1st_test`: channels `(0,1)`, `(2,3)`, `(4,5)`, `(6,7)`;
- `2nd_test` and `3rd_test`: channels `(0,)`, `(1,)`, `(2,)`, `(3,)`.

For every channel in a bearing group, the loader computes RMS, standard
deviation, fourth-moment kurtosis, and crest factor, then averages each scalar
across the group. The four resulting columns are `rms`, `std`, `kurtosis`, and
`crest_factor`. Channel groups can be overridden in the YAML configuration;
the resolved groups are saved in `dataset_summary.json` and the run manifest.

This contract intentionally avoids an implicit envelope filter, resampling, or
target-derived health index. Such transformations require a separate versioned
protocol and a new benchmark profile.

## Split and evaluation

The default split is condition-based: `1st_test` bearing entities are source,
`2nd_test` entities are validation, and `3rd_test` entities are held-out target
entities. Explicit entity lists are supported but must cover every extracted
bearing exactly once. Scaling is fitted on source bearing rows only. Windows,
MASE, source-standardized metrics, model selection, shift summaries, and
generalization gaps use the same cross-domain runner as the process and
turbine profiles.

The executable configuration is
`configs/ims_bearings_cross_domain.yaml`. A real result should only be called a
benchmark profile after the archive manifest is present, the raw-file audit
passes, and the cross-domain audit recomputes every prediction and metric.
