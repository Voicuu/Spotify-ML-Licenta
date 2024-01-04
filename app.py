from imports import *
from data_loading_and_cleaning import load_and_clean_data
from data_transformation import transform_data
from modeling import run_models

st.title('Spotify Popularity Prediction')
model_folder = 'trained_models/'

def train_and_save_models(df, model_folder):
    # Transform the data
    X_train_preprocessed, X_test_preprocessed, y_train, y_test, _, column_transformer = transform_data(df, [])
    
    # Run the models
    run_models(X_train_preprocessed, X_test_preprocessed, y_train, y_test)
    
    # Save the column transformer
    with open(os.path.join(model_folder, 'column_transformer.pkl'), 'wb') as f:
        pickle.dump(column_transformer, f)

def load_model_and_transformer(model_folder):
    best_model_name_path = os.path.join(model_folder, 'best_model_name.txt')

    # Check if the best_model_name file exists, otherwise train the models
    if not os.path.isfile(best_model_name_path):
        df = load_and_clean_data()
        train_and_save_models(df, model_folder)

    if os.path.isfile(best_model_name_path):
        with open(best_model_name_path, 'r') as f:
            best_model_name = f.read().strip()
        best_model_path = os.path.join(model_folder, f'{best_model_name}_model.pkl')
        column_transformer_path = os.path.join(model_folder, 'column_transformer.pkl')

        if os.path.isfile(best_model_path) and os.path.isfile(column_transformer_path):
            with open(best_model_path, 'rb') as f:
                best_model = pickle.load(f)
            with open(column_transformer_path, 'rb') as f:
                column_transformer = pickle.load(f)
            return best_model, column_transformer
    return None, None

def parse_artists(artist_string):
    # Check if artist_string is a string
    if isinstance(artist_string, str):
        try:
            # list
            if artist_string.startswith("["):
                artist_list = ast.literal_eval(artist_string)
            else:
                #single artist name
                artist_list = [artist_string.strip('"')]
            
            # Clean the artist's name
            cleaned_artist_list = [artist.strip().strip("'") for artist in artist_list]

            return ', '.join(cleaned_artist_list)
        except (ValueError, SyntaxError) as e:
            # If there's an error in conversion, return the original string
            return artist_string
    else:
        # If artist_string is not a string, just return it as is
        return artist_string

def predict_and_save_results(artists_list, model, column_transformer, model_folder):
    df = load_and_clean_data()
    _, _, _, _, cases_with_mode, _ = transform_data(df, artists_list)

    if cases_with_mode.empty:
        return pd.DataFrame()
    
    # Convert the list of artists in each row to a string
    cases_with_mode['artists'] = cases_with_mode['artists'].apply(lambda x: ', '.join(x))    
    cases_X = cases_with_mode.drop(['popularity_level', 'explicit', 'id', 'release_date'], axis=1, errors='ignore')
    cases_X_transformed = column_transformer.transform(cases_X)
    cases_with_mode['predicted_popularity'] = model.predict(cases_X_transformed)
    cases_with_mode_path = os.path.join(model_folder, 'cases_with_mode.csv')
    cases_with_mode.to_csv(cases_with_mode_path, index=False)
    return cases_with_mode

def on_form_submit():
    with st.spinner('Loading models and making predictions...'):
        df = load_and_clean_data()
        best_model, column_transformer = load_model_and_transformer(model_folder)

        if user_artists and best_model and column_transformer:
            all_individual_artist_predictions = []

            for i, artist in enumerate(user_artists):
                artist_data = df[df['artists'].apply(lambda x: artist in x)]
                if artist_data.empty:
                    st.error(f"Artist '{artist}' not found in the dataset. Check the artist name and try again.")
                    continue

                individual_artist_predictions = predict_and_save_results([artist], best_model, column_transformer, model_folder)
                if individual_artist_predictions is not None and not individual_artist_predictions.empty:
                    all_individual_artist_predictions.append(individual_artist_predictions)

            if all_individual_artist_predictions:
                updated_cases_with_mode = pd.concat(all_individual_artist_predictions, ignore_index=True)
                if {'artists', 'name', 'popularity', 'popularity_level', 'predicted_popularity'}.issubset(updated_cases_with_mode.columns):
                    st.write(updated_cases_with_mode[['artists', 'name', 'popularity', 'popularity_level', 'predicted_popularity']].head(100))
                else:
                    st.error("Expected columns are missing from the prediction results.")
            else:
                st.error("No predictions were made. Please check the artist names and try again.")
        else:
            st.error("Please select at least one artist to predict popularity.")



with st.form(key='user_input_form'):
    df = load_and_clean_data()

    all_artists = list(set(chain.from_iterable(df['artists'])))
    user_artists = st.multiselect('Enter artists', options=sorted(all_artists), key='user_artists')
    submit_button = st.form_submit_button(label='Predict Popularity')

if submit_button:
    on_form_submit()