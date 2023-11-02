from imports import *

def transform_data(df):
    # Convert milliseconds to minutes for better interpretability
    df["duration_mins"] = df["duration_ms"]/60000
    df.drop(columns="duration_ms", inplace=True)

    # Remove special characters from the 'artists' column using regex
    df["artists"] = df["artists"].str.replace("[\\[\\]']", "", regex=True)

    data = df.copy()

    # Mapping popularity into categorical levels
    data['popularity_level'] = pd.cut(data.popularity, bins=[-1, 30, 60, 100], labels=[1, 2, 3]).astype(int)

    artists = ['Drake', 'Lady Gaga', 'Taylor Swift', 'The Weeknd', 'Da Baby']
    
    # Create a list of indices corresponding to the artists above
    to_drop = data[data.artists.isin(artists)].index
    
    # Gather the test cases
    cases_with_mode = data.loc[to_drop].copy()
    
    # Remove the test cases from data
    data.drop(to_drop, inplace=True)

    # Drop columns only if they exist in the DataFrame
    columns_to_drop = ["popularity", "explicit", "id", "mode", "release_date", "artists", "name"]
    data.drop(columns=[col for col in columns_to_drop if col in data.columns], inplace=True)

    # Under-sample the majority class to balance the dataset and improve model performance
    data.drop(data[data['popularity_level']==2].index[:60000], inplace=True)
    data.drop(data[data['popularity_level']==1].index[:60000], inplace=True)

    # Check if data is empty after dropping rows
    if data.empty:
        raise ValueError("Data is empty after dropping rows.")

    y = data['popularity_level']
    X = data.drop(columns=['popularity_level'])
    
    # Check if X is empty after dropping the target column
    if X.empty:
        raise ValueError("Feature data (X) is empty after preprocessing.")
    
    X_train, X_test, y_train, y_test = split(X, y, test_size = 0.25, random_state = 42)

    # Check if training or test set is empty after splitting
    if X_train.empty or X_test.empty:
        raise ValueError("Training or test set is empty after splitting.")
    
    # Transform features by scaling specific features to a given range
    ctr = ColumnTransformer([
        ('minmax', MinMaxScaler(), ['year', 'tempo', 'duration_mins']),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['key'])
    ], remainder='passthrough')

    ctr.fit(X_train)
    X_train_preprocessed = ctr.transform(X_train)
    X_test_preprocessed = ctr.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, cases_with_mode, ctr
