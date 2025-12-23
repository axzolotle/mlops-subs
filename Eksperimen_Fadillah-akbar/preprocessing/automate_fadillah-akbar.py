import pandas as pd
from pathlib import Path

# PATH CONFIG
RAW_PATH = Path("Eksperimen_fadillah-akbar/AIML_Dataset.csv")
OUTPUT_PATH = Path("Eksperimen_fadillah-akbar/preprocessing/data_clean.csv")

# LOAD DATA
def load_data(path: Path, nrows : int) -> pd.DataFrame:
    return pd.read_csv(path, nrows=nrows)

# PREPROCESSING
def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    # One-hot encoding
    df = pd.get_dummies(df, columns=["type"], drop_first=True)

    # Drop duplicates
    df = df.drop_duplicates()

    # Frequency encoding
    df["dest_freq"] = df["nameDest"].map(df["nameDest"].value_counts())
    df["orig_freq"] = df["nameOrig"].map(df["nameOrig"].value_counts())

    # Drop identifier columns
    df = df.drop(columns=["nameDest", "nameOrig"])

    # Drop nulls (aman)
    df = df.dropna()

    return df

# MAIN
def main():
    df = load_data(RAW_PATH, nrows=100000)

    df_clean = preprocess(df)

    df_clean.to_csv(OUTPUT_PATH, index=False)


if __name__ == "__main__":
    main()
