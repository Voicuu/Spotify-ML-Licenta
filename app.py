from imports import *
from data_loading_and_cleaning import load_and_clean_data
from data_transformation import transform_data
from modeling import run_models

# Initialize the app and title
st.title('Spotify Popularity Prediction')

# The model folder where trained models and transformers are stored
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

# Text input for artist names
user_artists = st.text_input('Enter artists separated by comma', key='user_artists')

# Function to handle the Enter press on the text input
def on_text_enter():
    # Load the model and transformer, and train if necessary
    best_model, column_transformer = load_model_and_transformer(model_folder)

    if user_artists and best_model and column_transformer:
        # Split the user input into a list of artists
        artists_list = [artist.strip() for artist in user_artists.split(',')]
        # Get the updated predictions
        updated_cases_with_mode = predict_and_save_results(artists_list, best_model, column_transformer, model_folder)
        # Display the results
        updated_cases_with_mode.reset_index(drop=True, inplace=True)
        #updated_cases_with_mode.index += 1
        desired_columns = ['artists', 'name', 'popularity', 'popularity_level', 'predicted_popularity']
        st.write(updated_cases_with_mode[desired_columns].head(50))

# Function to predict and save results
def predict_and_save_results(artists_list, model, column_transformer, model_folder):
    df = load_and_clean_data()
    _, _, _, _, cases_with_mode, _ = transform_data(df, artists_list)
    cases_X = cases_with_mode.drop(['popularity_level', 'explicit', 'id', 'release_date'], axis=1, errors='ignore')
    cases_X_transformed = column_transformer.transform(cases_X)
    cases_with_mode['predicted_popularity'] = model.predict(cases_X_transformed)
    cases_with_mode_path = os.path.join(model_folder, 'cases_with_mode.csv')
    cases_with_mode.to_csv(cases_with_mode_path, index=False)
    return cases_with_mode

# Check if the callback should be triggered based on user input
if user_artists:
    on_text_enter()
    
