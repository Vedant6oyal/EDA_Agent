import pandas as pd

def load_snapshots(path='snapshots_2000.csv'):
    df = pd.read_csv(path)
    df = df[df['days_since_last_event'].notna()]
    return df
