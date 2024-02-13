from imports import *


def transform_data(df, artists):
    # Convert milliseconds to minutes for better interpretability
    df["duration_mins"] = df["duration_ms"] / 60000
    df.drop(columns="duration_ms", inplace=True)

    # Flag rows where artist matches user input
    df["artist_match"] = df["artists"].apply(
        lambda x: any(artist in x for artist in artists)
    )

    # Copy df to avoid altering original dataframe
    data = df.copy()

    # Mapping popularity into categorical levels
    data["popularity_level"] = pd.cut(
        data["popularity"], bins=[-1, 30, 60, 100], labels=[1, 2, 3]
    ).astype(int)

    # Separate the test cases before dropping unnecessary rows
    cases_with_mode = data[data["artist_match"]].copy()

    # Drop columns not needed for model training
    columns_to_drop = [
        "popularity",
        "explicit",
        "id",
        "mode",
        "release_date",
        "artists",
        "name",
    ]
    data.drop(
        columns=[col for col in columns_to_drop if col in data.columns], inplace=True
    )

    # Under-sample the majority class to balance the dataset
    data.drop(data[data["popularity_level"] == 2].index[:60000], inplace=True)
    data.drop(data[data["popularity_level"] == 1].index[:60000], inplace=True)

    # Split data into features and target
    y = data.popularity_level
    X = data.drop(columns=["popularity_level"])

    # Split data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    # Define and fit a column transformer for preprocessing
    ctr = ColumnTransformer(
        transformers=[
            ("minmax", MinMaxScaler(), ["duration_mins", "tempo", "year"]),
            ("onehot", OneHotEncoder(handle_unknown="ignore"), ["key"]),
        ],
        remainder="passthrough",
    )

    X_train_preprocessed = ctr.fit_transform(X_train)
    X_test_preprocessed = ctr.transform(X_test)

    return (
        X_train_preprocessed,
        X_test_preprocessed,
        y_train,
        y_test,
        cases_with_mode,
        ctr,
    )
