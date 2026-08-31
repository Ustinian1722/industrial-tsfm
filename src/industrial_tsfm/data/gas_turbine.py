from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cmapss import EntitySplits

FEATURE_COLUMNS = (
    "AT",
    "AP",
    "AH",
    "AFDP",
    "GTEP",
    "TIT",
    "TAT",
    "TEY",
    "CDP",
    "CO",
    "NOX",
)
YEARS = (2011, 2012, 2013, 2014, 2015)


@dataclass
class GasTurbineDataset:
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    splits: EntitySplits
    years: tuple[int, ...]

    def summary(self) -> dict[str, object]:
        return {
            "dataset": "uci_gas_turbine_co_nox",
            "feature_columns": list(self.feature_columns),
            "n_features": len(self.feature_columns),
            "rows": len(self.frame),
            "entities": int(self.frame["entity_id"].nunique()),
            "years": list(self.years),
            "splits": self.splits.as_dict(),
        }


def _read_year(path: Path, year: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing UCI Gas Turbine file: {path}")
    frame = pd.read_csv(path)
    frame.columns = [str(column).strip().upper() for column in frame.columns]
    aliases = {"NOX": "NOX", "NOX (MG/M3)": "NOX", "NOX_MG/M3": "NOX"}
    frame = frame.rename(columns={column: aliases.get(column, column) for column in frame.columns})
    missing = set(FEATURE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing Gas Turbine columns in {path}: {sorted(missing)}")
    values = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    if values.isna().any().any() or not np.isfinite(values.to_numpy()).all():
        raise ValueError(f"Gas Turbine file contains missing/non-finite values: {path}")
    values = values.astype(np.float64)
    values["cycle"] = np.arange(1, len(values) + 1, dtype=np.int64)
    values["year"] = year
    values["source"] = "uci_gas_turbine"
    values["entity_id"] = f"gas_turbine_year_{year}"
    return values


def load_gas_turbine(
    raw_dir: str | Path,
    train_years: tuple[int, ...] = (2011, 2012),
    validation_years: tuple[int, ...] = (2013,),
    test_years: tuple[int, ...] = (2014, 2015),
) -> GasTurbineDataset:
    all_years = tuple(dict.fromkeys((*train_years, *validation_years, *test_years)))
    if set(all_years) != set(YEARS):
        raise ValueError(f"The loader expects all UCI years {YEARS} exactly once")
    raw_dir = Path(raw_dir)
    frames = [_read_year(raw_dir / f"gt_{year}.csv", year) for year in all_years]
    frame = pd.concat(frames, ignore_index=True).sort_values(["entity_id", "cycle"], kind="stable")
    splits = EntitySplits(
        tuple(f"gas_turbine_year_{year}" for year in train_years),
        tuple(f"gas_turbine_year_{year}" for year in validation_years),
        tuple(f"gas_turbine_year_{year}" for year in test_years),
    )
    splits.assert_disjoint()
    return GasTurbineDataset(frame.reset_index(drop=True), FEATURE_COLUMNS, splits, all_years)
