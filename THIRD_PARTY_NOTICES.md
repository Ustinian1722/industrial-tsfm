# Third-party software, models, and data

The Apache-2.0 license in `LICENSE` applies to original project code and
project-authored configuration/documentation unless a file says otherwise.
It does not relicense third-party software, pretrained weights, or datasets.
The repository does not check in raw datasets, model checkpoints, or model
cache files.

## Python dependencies

Runtime and optional dependencies are declared in `pyproject.toml`. Their
licenses and notices remain governed by their respective distributions. The
main optional integrations are:

- NumPy, pandas, PyYAML, scikit-learn, and PyTorch;
- TimesFM and Chronos forecasting packages;
- Uni2TS/GluonTS for the Moirai-2 adapter;
- Transformers, Accelerate, and PEFT for the TimesFM LoRA adapter.

Installers should resolve and review the licenses of the exact versions used
in a release environment.

## Pretrained models

Model identifiers are recorded in the versioned configs and run manifests.
Weights are downloaded by the user from the upstream model service and are not
included in this repository. Follow the model card and checkpoint terms for
every use or redistribution:

- TimesFM: <https://huggingface.co/google/timesfm-2.5-200m-pytorch>
- TimesFM Transformers: <https://huggingface.co/google/timesfm-2.5-200m-transformers>
- Chronos: <https://huggingface.co/amazon/chronos-t5-tiny>
- Chronos-2: <https://huggingface.co/amazon/chronos-2>
- Moirai-2.0-R-small: <https://huggingface.co/Salesforce/moirai-2.0-R-small>

The current Moirai-2 model card states a CC BY-NC 4.0 model license. Treat
that checkpoint as non-commercial unless you have separate permission. This
restriction does not change the Apache-2.0 license of the original adapter
code.

## Datasets

Fetch scripts record source URLs, hashes, and dataset-specific notes in
`source_manifest.json`. They require explicit source acceptance where the
source boundary is material. Raw files are ignored by Git and must not be
added to commits or releases.

- NASA C-MAPSS: [NASA Dashlink description](https://c3.ndc.nasa.gov/dashlink/resources/139/)
  and [NASA catalog metadata](https://catalog.data.gov/dataset/c-mapss-aircraft-engine-simulator-data)
- UCI Gas Turbine: [UCI dataset page](https://archive.ics.uci.edu/dataset/551/gas+turbine+co+and+nox+emission+data+set)
  and the documented [transport mirror](https://github.com/skforecast/skforecast-datasets)
- Tennessee Eastman: [University of Washington archive](https://depts.washington.edu/control/LARRY/TE/download.html)
- NASA/IMS Bearings: [NASA catalog entry](https://catalog.data.gov/dataset/ims-bearings)
  and [NASA PCoE repository](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)

The Tennessee Eastman archive page does not state an SPDX license in the
current protocol, so keep those raw files local and verify permissions before
redistribution. The IMS archive is large and remains an opt-in local download;
its feature-level benchmark contract is documented in
`docs/ims_bearings_protocol.md`.

## Release rule

Before publishing a release, regenerate dependency/license metadata for the
exact environment, review every upstream model and dataset term, and verify
that the release contains no raw data, downloaded weights, secrets, or private
experiment artifacts.
