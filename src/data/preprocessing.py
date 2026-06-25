from __future__ import annotations

import pandas as pd

from .load_data import load_data

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw churn dataset before feature engineering.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with ID column removed and TotalCharges converted to numeric.
    """

    data = df.copy()

    data.columns = data.columns.str.strip()

    id_columns = ["CustomerID", "customerID", "customer_id"]
    data = data.drop(columns=[col for col in id_columns if col in data.columns])

    data["TotalCharges"] = (
        data["TotalCharges"]
        .replace(" ", pd.NA)
        .pipe(pd.to_numeric, errors="coerce")
    )

    data = data.dropna(subset=["TotalCharges"]).reset_index(drop=True)

    return data