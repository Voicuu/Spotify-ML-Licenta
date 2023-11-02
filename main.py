from imports import *
from data_loading_and_cleaning import load_and_clean_data
from data_visualization import visualize_data
from data_transformation import transform_data
from modeling import run_models

def display_results(results):
    rows = [(res[0], res[1], res[3]) for res in results]
    tab = tabulate(rows, headers=['Algorithm', 'Accuracy', 'F1 Score'], tablefmt='fancy_grid')
    print(tab)

def display_example_cases(df, cases_with_mode, best_model, ctr):
    # Sample and reset index
    cases_mix = cases_with_mode.sample(frac=1.0, random_state=42).reset_index(drop=True)

    # Prepare data for prediction
    cases_X = cases_mix.drop(['popularity_level', 'explicit', 'id', 'release_date'], axis=1, errors='ignore')

    if cases_X.empty:
        raise ValueError("cases_X is empty, can't transform it.")

    # Transform cases_X using the ColumnTransformer fitted on the training data
    cases_X_transformed = ctr.transform(cases_X)

    # Predict and create a DataFrame with matching indices
    cases_pred = pd.DataFrame(best_model.predict(cases_X_transformed), columns=['predicted_popularity'])

    # Concatenate predictions with original data
    res = pd.concat([cases_mix, cases_pred], axis=1)

    # Display the desired columns
    desired_columns = ['artists', 'name', 'popularity', 'popularity_level', 'predicted_popularity']
    print(tabulate(res[desired_columns].head(10), headers='keys', tablefmt='psql'))

def main():
    df = load_and_clean_data()
    #print(df.columns)
    #visualize_data(df)
    X_train, X_test, y_train, y_test, cases_with_mode, ctr = transform_data(df)

    results = run_models(X_train, X_test, y_train, y_test)
    display_results(results)

    # Choose the best performing model for displaying example cases
    best_model = max(results, key=lambda x: x[1])[2] 
    display_example_cases(df, cases_with_mode, best_model, ctr)

if __name__ == "__main__":
    main()