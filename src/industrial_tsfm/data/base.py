from __future__ import annotations

from typing import Protocol

import pandas as pd

from .cmapss import EntitySplits


class IndustrialForecastDataset(Protocol):
    """Common contract consumed by the cross-domain benchmark runner."""

    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    splits: EntitySplits

    def summary(self) -> dict[str, object]:
        ...
