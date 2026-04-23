from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dc_reif.config import ProjectConfig
from dc_reif.paths import ProjectPaths


def make_sample_king_county_df(n_rows: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2014-05-01", periods=n_rows, freq="D")
    sqft_living = rng.integers(800, 4200, size=n_rows)
    grade = rng.integers(5, 12, size=n_rows)
    lat = 47.2 + rng.random(n_rows) * 0.6
    lon = -122.5 + rng.random(n_rows) * 0.5
    house_age = rng.integers(1, 80, size=n_rows)
    yr_built = 2015 - house_age
    yr_renovated = np.where(rng.random(n_rows) > 0.8, yr_built + rng.integers(5, 40, size=n_rows), 0)
    price = (
        sqft_living * 260
        + grade * 18000
        + (lat - 47.2) * 100000
        + rng.normal(0, 25000, size=n_rows)
    ).clip(min=80000)

    return pd.DataFrame(
        {
            "id": [f"{1000000 + idx}" for idx in range(n_rows)],
            "date": dates.strftime("%Y-%m-%d"),
            "price": price.round(0),
            "bedrooms": rng.integers(1, 6, size=n_rows),
            "bathrooms": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], size=n_rows),
            "sqft_living": sqft_living,
            "sqft_lot": rng.integers(1500, 12000, size=n_rows),
            "floors": rng.choice([1.0, 1.5, 2.0, 2.5, 3.0], size=n_rows),
            "waterfront": rng.integers(0, 2, size=n_rows),
            "view": rng.integers(0, 5, size=n_rows),
            "condition": rng.integers(2, 5, size=n_rows),
            "grade": grade,
            "sqft_above": (sqft_living * rng.uniform(0.7, 1.0, size=n_rows)).round(0),
            "sqft_basement": (sqft_living * rng.uniform(0.0, 0.3, size=n_rows)).round(0),
            "yr_built": yr_built,
            "yr_renovated": yr_renovated,
            "zipcode": rng.choice(["98001", "98002", "98003", "98004"], size=n_rows),
            "lat": lat,
            "long": lon,
            "sqft_living15": (sqft_living * rng.uniform(0.8, 1.2, size=n_rows)).round(0),
            "sqft_lot15": rng.integers(1500, 12000, size=n_rows),
        }
    )


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    return make_sample_king_county_df()


@pytest.fixture
def temp_config(tmp_path: Path) -> ProjectConfig:
    project_root = tmp_path
    paths = ProjectPaths.from_root(project_root).ensure()
    return ProjectConfig(
        project_root=project_root,
        paths=paths,
        data_url="",
        data_filename="kc_house_data.csv",
        use_aria2=False,
        force_download=False,
    )
