from __future__ import annotations

from pathlib import Path
from typing import Any

from .base import IndustrialForecastDataset
from .gas_turbine import load_gas_turbine
from .ims_bearings import load_ims_bearings
from .tennessee_eastman import load_tennessee_eastman


def load_cross_domain_dataset(
    name: str,
    raw_dir: str | Path,
    config: dict[str, Any],
) -> IndustrialForecastDataset:
    """Load a dataset using the common cross-domain frame/split contract."""
    name = str(name).lower()
    if name == "uci_gas_turbine":
        return load_gas_turbine(
            raw_dir,
            train_years=tuple(int(value) for value in config.get("train_years", [2011, 2012])),
            validation_years=tuple(int(value) for value in config.get("validation_years", [2013])),
            test_years=tuple(int(value) for value in config.get("test_years", [2014, 2015])),
        )
    if name == "tennessee_eastman":
        return load_tennessee_eastman(
            raw_dir,
            train_disturbances=tuple(
                int(value) for value in config.get("train_disturbances", range(1, 9))
            ),
            validation_disturbances=tuple(
                int(value) for value in config.get("validation_disturbances", [9, 10])
            ),
            test_disturbances=tuple(
                int(value) for value in config.get("test_disturbances", range(11, 16))
            ),
            feature_set=str(config.get("feature_set", "measured")),
        )
    if name == "ims_bearings":
        return load_ims_bearings(
            raw_dir,
            subsets=tuple(str(value) for value in config.get("subsets", ["1st_test", "2nd_test", "3rd_test"])),
            file_interval_seconds=int(config.get("file_interval_seconds", 600)),
            max_files_per_subset=(
                int(config["max_files_per_subset"])
                if config.get("max_files_per_subset") is not None
                else None
            ),
            config=config,
        )
    raise ValueError(f"Unsupported cross-domain dataset: {name}")
