import pandas as pd

from src.data_utils import CleanConfig, clean_dataframe


def test_clean_dataframe_imputes_missing():
    df = pd.DataFrame(
        {
            "age": [50, None],
            "sex": [1, 0],
            "cp": ["3", None],
            "target": [0, 1],
        }
    )
    cleaned = clean_dataframe(df, CleanConfig())
    assert cleaned["age"].isna().sum() == 0
    assert cleaned["cp"].isna().sum() == 0


def test_clean_dataframe_binary_target():
    df = pd.DataFrame(
        {
            "age": [60, 55],
            "sex": [1, 0],
            "cp": [2, 1],
            "target": [0, 2],
        }
    )
    cleaned = clean_dataframe(df, CleanConfig())
    assert set(cleaned["target"].unique()) == {0, 2}
