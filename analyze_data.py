import pandas as pd

try:
    df = pd.read_csv('dataset/K4train.csv')
    print("Columns:", df.columns.tolist())
    for col in df.columns:
        if col != 'Timestamp':
            print(f"Unique values in {col} ({len(df[col].unique())}): {df[col].unique()[:10]}")
except Exception as e:
    print(e)
