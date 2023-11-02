from imports import *

def load_and_clean_data():
    df = pd.read_csv('data.csv')
    df = df.drop(["explicit", "id", "mode", "release_date"], axis=1)
    
    # Drop duplicates based on 'artists' and 'name'
    df = df.drop_duplicates(subset=['artists', 'name'], keep='first')
    
    # Fill NaN values for numeric columns only
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    return df
