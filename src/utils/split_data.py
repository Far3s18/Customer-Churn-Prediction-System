from __future__ import annotations

import pandas as pd

from sklearn.model_selection import train_test_split

from src.data.load_data import load_data


def split_data(
    data_path: str = "/telco-churn/data/processed/processed_telco_churn.csv",
    test_size: float = 0.10,
    val_size: float = 0.10,
    random_state: int = 41,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Split the processed Telco churn dataset into train, validation, and test sets.

    Returns
    -------
    tuple
        x_train, x_val, x_test, y_train, y_val, y_test
    """

    df = load_data(data_path)

    if "Churn" not in df.columns:
        raise ValueError("Target column 'Churn' not found in the dataset.")

    x = df.drop(columns=["Churn"])
    y = df["Churn"]

    x_train, x_temp, y_train, y_temp = train_test_split(
        x,
        y,
        test_size=val_size + test_size,
        random_state=random_state,
        stratify=y,
    )

    relative_test_size = test_size / (val_size + test_size)

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=relative_test_size,
        random_state=random_state,
        stratify=y_temp,
    )

    return x_train, x_val, x_test, y_train, y_val, y_test