from __future__ import annotations

from pathlib import Path
import pandas as pd


DATA_DIR = Path(__file__).parent / "data"


def load_sources() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "sources.csv")


def load_parameters() -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / "parameters.csv")


def source_for_parameter(parameter_id: str) -> pd.DataFrame:
    params = load_parameters()
    sources = load_sources()

    subset = params[params["parameter_id"] == parameter_id].copy()
    if subset.empty:
        return subset

    return subset.merge(
        sources,
        on="source_id",
        how="left",
        suffixes=("_parameter", "_source"),
    )


def get_parameter(parameter_id: str) -> dict:
    params = load_parameters()
    row = params[params["parameter_id"] == parameter_id]

    if row.empty:
        raise KeyError(f"Unknown parameter_id: {parameter_id}")

    return row.iloc[0].to_dict()
