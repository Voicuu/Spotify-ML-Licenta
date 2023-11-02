from imports import *

def run_model(model, X_train, y_train, X_test, y_test, alg_name):
    try:
        start_time = time()  # Record the start time
        
        # Creating a pipeline with SMOTE and the model
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('model', model)
        ])

        # Fit model
        pipeline.fit(X_train, y_train)
        
        fit_time = time()  # Record the time after fitting the model
        
        # Predict and evaluate
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
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

        return (alg_name, accuracy, model, f1)

    except Exception as e:
        print(f"An error occurred while running the model {alg_name}: {e}")
        return None

def hyperparameter_tuning(model, params, X_train, y_train):
    try:
        random_search = RandomizedSearchCV(model, params, n_iter=10, cv=3, n_jobs=-1, random_state=42, scoring='accuracy', verbose=2)
        random_search.fit(X_train, y_train)
        return random_search.best_estimator_

    except Exception as e:
        print(f"An error occurred during hyperparameter tuning: {e}")
        return None

def run_models(X_train, X_test, y_train, y_test):
    results = []

    # Decision Tree
    #dt_params = {
    #    'max_depth': [None, 10, 20, 30, 50],
    #    'min_samples_split': [2, 5, 10, 20],
    #    'min_samples_leaf': [1, 2, 4, 6],
    #    'max_features': [None, 'sqrt', 'log2'],
    #    'criterion': ['gini', 'entropy'],
    #    'class_weight': [None, 'balanced']
    #}
    #dt_best = hyperparameter_tuning(DecisionTreeClassifier(), dt_params, X_train, y_train)
    #if dt_best:
    #    results.append(run_model(dt_best, X_train, y_train, X_test, y_test, "Decision Tree"))

    ## SVM
    #svm_params = {'C': [0.1, 1, 10], 'gamma': [1, 0.1], 'kernel': ['rbf', 'poly']}
    #svm_best = hyperparameter_tuning(SVC(), svm_params, X_train, y_train)
    #results.append(run_model(svm_best, X_train, y_train, X_test, y_test, "SVM""))
#
    ## KNeighborsClassifier
    #knn_params = {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
    #knn_best = hyperparameter_tuning(KNeighborsClassifier(), knn_params, X_train, y_train)
    #results.append(run_model(knn_best, X_train, y_train, X_test, y_test, "KNeighborsClassifier"""))
#
    ## LogisticRegression
    #lr_params = {'C': [0.1, 1, 10]}
    #lr_best = hyperparameter_tuning(LogisticRegression(), lr_params, X_train, y_train)
    #results.append(run_model(lr_best, X_train, y_train, X_test, y_test, "Logistic Regression""))
#
    #AdaBoostClassifier
    ada_params = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    ada_best = hyperparameter_tuning(AdaBoostClassifier(), ada_params, X_train, y_train)
    results.append(run_model(ada_best, X_train, y_train, X_test, y_test, "AdaBoostClassifier"))

    #Random Forest with hyperparameter tuning
    rf_params = {
        'n_estimators': [100, 200, 300, 400],
        'max_depth': [None, 20, 30, 40],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt'],
        'class_weight': [None, 'balanced']
    }
    rf_best = hyperparameter_tuning(RandomForestClassifier(), rf_params, X_train, y_train)
    if rf_best:
        results.append(run_model(rf_best, X_train, y_train, X_test, y_test, "Random Forest"))

    return results