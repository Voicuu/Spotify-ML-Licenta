from imports import *

def transform_data(df,artists):
    # Filter data based on user input artists
    #artists = input("Enter artists separated by comma: ").split(',')
    #artists = [artist.strip() for artist in artists]

    # Convert milliseconds to minutes for better interpretability
    df["duration_mins"] = df["duration_ms"]/60000
    df.drop(columns="duration_ms", inplace=True)

    # Convert stringified lists in the 'artists' column to actual lists
    df['artists'] = df['artists'].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Filter data based on user input artists
    df['artist_match'] = df['artists'].apply(lambda artist_list: any(artist in artist_list for artist in artists))
    data = df[df['artist_match']].copy()

    # Mapping popularity into categorical levels
    data['popularity_level'] = pd.cut(data['popularity'], bins=[-1, 30, 60, 100], labels=[1, 2, 3]).astype(int)
    
    #artists = ['Drake', 'Lady Gaga', 'Taylor Swift', 'The Weeknd', 'Da Baby']
    
    # Ensure these columns are not in the list to be dropped
    columns_to_drop = ["explicit", "id", "mode", "release_date", 'artist_match']
    columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=columns_to_drop, inplace=True)

    # Under-sample the majority class to balance the dataset and improve model performance
    data_majority = data[data['popularity_level'] == 2]
    data_minority = data[data['popularity_level'] != 2]

    # Perform downsampling only if it's possible
    if data_majority.shape[0] > data_minority.shape[0]:
        # Proceed with downsampling
        data_majority_downsampled = data_majority.sample(n=data_minority.shape[0], random_state=42)
        data_balanced = pd.concat([data_majority_downsampled, data_minority])
    else:
        # If downsampling not needed, just concatenate the data as is
        data_balanced = pd.concat([data_majority, data_minority])

    # Separate the features and the target variable
    y = data_balanced['popularity_level']
    X = data_balanced.drop('popularity_level', axis=1)

    # Check if we have enough samples to perform the split
    if len(X) < 4:
        print(f"Not enough records to perform a train/test split for artists: {artists}")
        return None, None, None, None, pd.DataFrame(), None

    # Perform the split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define the transformation pipeline
    column_transformer = ColumnTransformer([
        ('minmax', MinMaxScaler(), ['year', 'tempo', 'duration_mins']),
        ('categorical', OneHotEncoder(handle_unknown='ignore'), ['key'])
    ], remainder='passthrough')

    column_transformer.fit(X_train)
    X_train_preprocessed = column_transformer.transform(X_train)
    X_test_preprocessed = column_transformer.transform(X_test)

    return X_train_preprocessed, X_test_preprocessed, y_train, y_test, data_balanced, column_transformer