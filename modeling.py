from imports import *


def run_model(model, X_train, y_train, X_test, y_test, alg_name):
    try:
        print(f"Starting training for {alg_name}.")
        start_time = time()  # Record the start time

        # Creating a pipeline with SMOTE and the model
        pipeline = ImbPipeline([("smote", SMOTE(random_state=42)), ("model", model)])

        # Fit model
        pipeline.fit(X_train, y_train)

        fit_time = time()  # Record the time after fitting the model

        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        Cr = classification_report(y_test, y_pred)

        evaluation_time = time()  # Record the time after evaluation

        # Display the results
        print(f"--- Model: {alg_name} ---")
        print(f"Model fitting took {fit_time - start_time:.2f} seconds.")
        print(f"Model evaluation took {evaluation_time - fit_time:.2f} seconds.")
        print(f"Total time: {evaluation_time - start_time:.2f} seconds.")
        print(f"Accuracy on Test Set: {accuracy:.2f}")
        print(f"F1 Score on Test Set: {f1:.2f}")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(Cr)
        print("-------------------------------\n")

        # Ensure the model is a pipeline that can be used for predictions
        if isinstance(pipeline, ImbPipeline):
            return (alg_name, accuracy, pipeline, f1)
        else:
            print(f"{alg_name} is not a valid pipeline.")
            return None
    except Exception as e:
        print(f"An error occurred while running the model {alg_name}: {e}")
        return None


def hyperparameter_tuning(model, params, X_train, y_train):
    try:
        random_search = RandomizedSearchCV(
            model,
            params,
            n_iter=10,
            cv=3,
            n_jobs=-1,
            random_state=42,
            scoring="accuracy",
            verbose=2,
        )
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

    except Exception as e:
        print(f"An error occurred during hyperparameter tuning: {e}")
        return None


def run_models(X_train, X_test, y_train, y_test):
    print("Starting the model training process...")
    results = []

    model_folder = "trained_models/"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    # Decision Tree
    dt_model_path = os.path.join(model_folder, "dt_model.pkl")
    if os.path.isfile(dt_model_path):
        with open(dt_model_path, "rb") as f:
            dt_best = pickle.load(f)
    else:
        dt_params = {
            "max_depth": [10, 20, 30, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2"],
            "criterion": ["gini", "entropy"],
        }
        dt_best = hyperparameter_tuning(
            DecisionTreeClassifier(), dt_params, X_train, y_train
        )
        if dt_best:
            with open(dt_model_path, "wb") as f:
                pickle.dump(dt_best, f)
    if isinstance(dt_best, DecisionTreeClassifier):
        results.append(
            run_model(dt_best, X_train, y_train, X_test, y_test, "DecisionTree")
        )

    # # SVM
    # this one takes too long to run omg
    # svm_model_path = os.path.join(model_folder, "svm_model.pkl")
    # if os.path.isfile(svm_model_path):
    #     with open(svm_model_path, "rb") as f:
    #         svm_best = pickle.load(f)
    # else:
    #     svm_params = {
    #         "C": [0.1, 1, 10, 100],
    #         "gamma": ["scale", 1, 0.1, 0.01],
    #         "kernel": ["rbf", "poly"],
    #     }
    #     svm_best = hyperparameter_tuning(
    #         SVC(probability=True), svm_params, X_train, y_train
    #     )
    #     if svm_best:
    #         with open(svm_model_path, "wb") as f:
    #             pickle.dump(svm_best, f)
    # if isinstance(svm_best, SVC):
    #     results.append(run_model(svm_best, X_train, y_train, X_test, y_test, "SVM"))
    ## KNeighborsClassifier
    # knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    # knn_best = hyperparameter_tuning(KNeighborsClassifier(), knn_params, X_train, y_train)
    # results.append(run_model(knn_best, X_train, y_train, X_test, y_test, "KNeighborsClassifier"""))
    #
    ## LogisticRegression
    # lr_params = {'C': [0.1, 1, 10]}
    # lr_best = hyperparameter_tuning(LogisticRegression(), lr_params, X_train, y_train)
    # results.append(run_model(lr_best, X_train, y_train, X_test, y_test, "Logistic Regression""))
    #

    # AdaBoostClassifier
    ada_model_path = os.path.join(model_folder, "ada_model.pkl")
    if os.path.isfile(ada_model_path):
        with open(ada_model_path, "rb") as f:
            ada_best = pickle.load(f)
    else:
        ada_params = {"n_estimators": [50, 100, 200], "learning_rate": [0.01, 0.1, 1]}
        ada_best = hyperparameter_tuning(
            AdaBoostClassifier(), ada_params, X_train, y_train
        )
        if ada_best:
            with open(ada_model_path, "wb") as f:
                pickle.dump(ada_best, f)
    if isinstance(ada_best, AdaBoostClassifier):
        results.append(
            run_model(ada_best, X_train, y_train, X_test, y_test, "AdaBoostClassifier")
        )

    # Random Forest with hyperparameter tuning
    rf_model_path = os.path.join(model_folder, "rf_model.pkl")
    if os.path.isfile(rf_model_path):
        with open(rf_model_path, "rb") as f:
            rf_best = pickle.load(f)
    else:
        rf_params = {
            "n_estimators": [100, 200, 300, 400],
            "max_depth": [None, 20, 30, 40],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt"],
            "class_weight": [None, "balanced"],
        }
        rf_best = hyperparameter_tuning(
            RandomForestClassifier(), rf_params, X_train, y_train
        )
        if rf_best:
            with open(rf_model_path, "wb") as f:
                pickle.dump(rf_best, f)
    if isinstance(rf_best, RandomForestClassifier):
        results.append(
            run_model(rf_best, X_train, y_train, X_test, y_test, "Random Forest")
        )

    # Save the results and the models
    results_df = pd.DataFrame(
        results, columns=["Algorithm", "Accuracy", "Model", "F1 Score"]
    )
    results_csv_path = os.path.join(model_folder, "results.csv")
    results_df.to_csv(results_csv_path, index=False)

    # Determine the best model based on your criteria (e.g., accuracy)
    best_model_entry = max(results, key=lambda x: x[1])
    best_model_name, best_model_accuracy, best_model_pipeline, best_model_f1 = (
        best_model_entry
    )

    # Save the best model name
    best_model_name_path = os.path.join(model_folder, "best_model_name.txt")
    with open(best_model_name_path, "w") as f:
        f.write(best_model_name)

    # Save each model as part of a pipeline
    for alg_name, _, model_pipeline, _ in results:
        model_path = os.path.join(model_folder, f"{alg_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_pipeline, f)

    print(
        f"Finished fitting models. Best model: {best_model_name} with accuracy: {best_model_accuracy}"
    )
    return results
