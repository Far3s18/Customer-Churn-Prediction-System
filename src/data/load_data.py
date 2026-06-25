import os
import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load a CSV dataset from the given file path to DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        Loaded dataset as a pandas DataFrame.
    """

    if not os.path.exists(file_path):
        raise FileExistsError(f"File not found: {file_path}")

    return pd.read_csv(file_path)