from __future__ import annotations

import pandas as pd

from pathlib import Path

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform the cleaned churn dataset into model-ready features.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned churn dataset.

    Returns
    -------
    pd.DataFrame
        Numeric feature matrix with encoded target and categorical variables.
    """

    data = df.copy()

    data.columns = data.columns.str.strip()

    data["Churn"] = data["Churn"].map({"No": 0, "Yes": 1})

    categorical_cols = data.select_dtypes(include="object").columns.tolist()

    data = pd.get_dummies(
        data=data,
        columns=categorical_cols,
        drop_first=True,
        dtype=int
    )
    
    output_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "processed_telco_churn.csv"
    data.to_csv(output_file, index=False)

    return data