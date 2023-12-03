from imports import *
from data_loading_and_cleaning import load_and_clean_data
from data_visualization import visualize_data
from data_transformation import transform_data
from modeling import run_models
from tabulate import tabulate

def display_results(filepath):
    results_df = pd.read_csv(filepath)
    print(tabulate(results_df, headers='keys', tablefmt='psql'))

def display_example_cases(cases_with_mode_path, best_model_path, column_transformer_path):
    with open(column_transformer_path, 'rb') as f:
        ctr = pickle.load(f)
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)
    cases_with_mode = pd.read_csv(cases_with_mode_path)
    cases_X_transformed = ctr.transform(cases_with_mode.drop(['popularity_level', 'explicit', 'id', 'release_date'], axis=1, errors='ignore'))
    cases_with_mode['predicted_popularity'] = best_model.predict(cases_X_transformed)
    desired_columns = ['artists', 'name', 'popularity', 'popularity_level', 'predicted_popularity']
    print(tabulate(cases_with_mode[desired_columns].head(10), headers='keys', tablefmt='psql'))

def main():
    model_folder = 'trained_models/'
    cases_with_mode_path = os.path.join(model_folder, 'cases_with_mode.csv')
    best_model_name_path = os.path.join(model_folder, 'best_model_name.txt')
    column_transformer_path = os.path.join(model_folder, 'column_transformer.pkl')

    # Load and display results if available
    results_csv_path = os.path.join(model_folder, 'results.csv')
    if os.path.isfile(results_csv_path):
        display_results(results_csv_path)
    
    # Load and display example cases if available
    if os.path.isfile(cases_with_mode_path) and os.path.isfile(best_model_name_path):
        with open(best_model_name_path, 'r') as f:
            best_model_name = f.read().strip()
        best_model_path = os.path.join(model_folder, f'{best_model_name}_model.pkl')
        if os.path.isfile(best_model_path) and os.path.isfile(column_transformer_path):
            display_example_cases(cases_with_mode_path, best_model_path, column_transformer_path)
        else:
            print("Model or transformer file not found. Please run the web app to generate the model.")
    else:
        print("Cases with mode file not found. Please add artists in the web app.")

if __name__ == "__main__":
    main()
