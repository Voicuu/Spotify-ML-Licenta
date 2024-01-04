from imports import *

def load_and_clean_data():
    df = pd.read_csv('data.csv')
    df = df.drop(["explicit", "id", "mode", "release_date"], axis=1)
    
    # Drop duplicates based on 'artists' and 'name'
    df = df.drop_duplicates(subset=['artists', 'name'], keep='first')
    
    # Fill NaN values for numeric columns only
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # Convert the string representation of lists into actual lists
    df['artists'] = df['artists'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    
    return df