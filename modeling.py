from imports import *
import plotly.io as pio

def run_model(model, X_train, y_train, X_test, y_test, alg_name):
    try:
        print(f"Starting training for {alg_name}.")
        start_time = time.time()  # Record the start time

        if alg_name in [
            "XGBoost",
            "LightGBM",
        ]:
            resampling = ADASYN(random_state=42)
        else:
            resampling = SMOTE(random_state=42)

        pipeline = ImbPipeline([("resample", resampling), ("model", model)])

        pipeline.fit(X_train, y_train)

        fit_time = time.time()  # Record the time after fitting the model

        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        Cr = classification_report(y_test, y_pred)

        evaluation_time = time.time()  # Record the time after evaluation

        # results
        print(f"--- Model: {alg_name} ---")
        print(f"Model fitting took {fit_time - start_time:.2f} seconds.")
        print(f"Model evaluation took {evaluation_time - fit_time:.2f} seconds.")
        print(f"Total time: {evaluation_time - start_time:.2f} seconds.")
        print(f"Accuracy on Test Set: {accuracy:.5f}")
        print(f"F1 Score on Test Set: {f1:.5f}")
        print("Confusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(Cr)
        print("-------------------------------\n")

        if isinstance(pipeline, ImbPipeline):
            return (alg_name, accuracy, pipeline, f1)
        else:
            print(f"{alg_name} is not a valid pipeline.")
            return None
    except Exception as e:
        print(f"An error occurred while running the model {alg_name}: {e}")
        return None


def hyperparameter_tuning(model_class, X_train, y_train, X_test, y_test, n_trials=100):
    def objective(trial):
        if model_class == XGBClassifier:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }
        elif model_class == LGBMClassifier:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 20, 300),
                'min_child_samples': trial.suggest_int('min_child_samples', 20, 100),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 20, 100),
                'min_gain_to_split': trial.suggest_float('min_gain_to_split', 0, 15),
                'verbose': -1,
            }
            if params['max_depth'] == 20:
                params['max_depth'] = -1
        elif model_class == DecisionTreeClassifier:
            params = {
                'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
                'max_depth': trial.suggest_int('max_depth', 1, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
            }
        elif model_class == AdaBoostClassifier:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 2.0),
            }
        elif model_class == RandomForestClassifier:
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 200),
                'max_depth': trial.suggest_int('max_depth', 3, 20),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
                'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            }
#        elif model_class == SVC:
#            params = {
#                'C': trial.suggest_loguniform('C', 1e-3, 1e2),
#                'kernel': trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf', 'sigmoid']),
#                'gamma': trial.suggest_categorical('gamma', ['scale', 'auto']),
#            }
        else:
            raise NotImplementedError("This class is not yet implemented for tuning")


        model = model_class(**params)
        resampler = ADASYN(random_state=42) if model_class in [XGBClassifier, LGBMClassifier] else SMOTE(random_state=42)
        pipeline = ImbPipeline(steps=[('resample', resampler), ('model', model)])
        
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        return accuracy

    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial), n_trials=n_trials)

    best_params = study.best_params
    best_model = model_class(**best_params)
    best_model.fit(X_train, y_train)

    return best_model, study

def visualize_study(study, model_name, output_folder):
    try:
        model_output_folder = os.path.join(output_folder, model_name)
        if not os.path.exists(model_output_folder):
            os.makedirs(model_output_folder)

        # Optimization History
        fig1 = ov.plot_optimization_history(study)
        pio.write_image(fig1, os.path.join(model_output_folder, f"{model_name}_optimization_history.png"))
        
        # Param Importance
        fig2 = ov.plot_param_importances(study)
        pio.write_image(fig2, os.path.join(model_output_folder, f"{model_name}_param_importance.png"))

        print(f"Visualizations saved for {model_name}.")
    except ImportError:
        print("Optuna visualization is not available. Ensure you have matplotlib installed.")
    except Exception as e:
        print(f"An error occurred while visualizing the study for {model_name}: {e}")

def run_models(X_train, X_test, y_train, y_test):
    print("Starting the model training process...")
    results = []

    model_folder = "trained_models/"
    output_folder = "visualizations/"

    if not os.path.exists(model_folder):
        os.makedirs(model_folder)

    models_to_tune_optuna = {
        "XGBoost": (XGBClassifier(use_label_encoder=False, eval_metric="logloss"), 100),
        "LightGBM": (LGBMClassifier(), 100),
        "DecisionTree": (DecisionTreeClassifier(), 100),
        "AdaBoost": (AdaBoostClassifier(), 50),
        "RandomForest": (RandomForestClassifier(), 50),
        #"SVM": (SVC(probability=True), 50), 
    }

    studies = {}

    for model_name, (model, n_trials) in models_to_tune_optuna.items():
        print(f"\nWorking on {model_name}...")
        model_path = os.path.join(model_folder, f"{model_name.lower()}_model.pkl")

        if os.path.isfile(model_path):
            print(f"Loading saved {model_name} model...")
            with open(model_path, "rb") as f:
                best_model = pickle.load(f)
        else:
            print(f"Tuning hyperparameters for {model_name}...")
            best_model, study = hyperparameter_tuning(model.__class__, X_train, y_train, X_test, y_test, n_trials)
            with open(model_path, "wb") as f:
                pickle.dump(best_model, f)
            studies[model_name] = study

        result = run_model(best_model, X_train, y_train, X_test, y_test, model_name)
        if result is not None:
            results.append(result)

    results_df = pd.DataFrame(results, columns=["Algorithm", "Accuracy", "Model", "F1 Score"])
    results_csv_path = os.path.join(model_folder, "results.csv")
    results_df.to_csv(results_csv_path, index=False)

    best_model_entry = max(results, key=lambda x: x[1])
    best_model_name, best_model_accuracy, best_model_pipeline, best_model_f1 = best_model_entry

    best_model_name_path = os.path.join(model_folder, "best_model_name.txt")
    with open(best_model_name_path, "w") as f:
        f.write(best_model_name)

    for alg_name, _, model_pipeline, _ in results:
        model_path = os.path.join(model_folder, f"{alg_name}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_pipeline, f)

    for model_name, study in studies.items():
        print(f"Visualizations for {model_name}:")
        visualize_study(study, model_name,  output_folder)

    print("Finished fitting and tuning models.")
    return result