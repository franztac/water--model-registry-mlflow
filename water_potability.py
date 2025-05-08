import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import mlflow
from mlflow.models import infer_signature


mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("water_potability_hp")

# Load dataset
data = pd.read_csv(r"C:\Users\UTENTE\Git Repos\water_potability.csv")

# Split data into training and test sets
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


# Function to fill missing values with median
def fill_missing_with_median(df_original):
    df = df_original.copy()
    for column in df.columns:
        if df[column].isnull().any():
            median_value = df[column].median()
            df[column] = df[column].fillna(median_value)
    return df


# Fill missing values with median for training and test sets
train_processed_data = fill_missing_with_median(train_data)
test_processed_data = fill_missing_with_median(test_data)


# Prepare training data
# X_train = train_processed_data.iloc[:, 0:-1].values
# y_train = train_processed_data.iloc[:, -1].values

X_train = train_processed_data.drop(columns=["Potability"])
y_train = train_processed_data.Potability


#  Define the model and parameter distribution for RandomizedSearchCV
rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "max_depth": [None, 4, 5, 6, 10],
}

# Perform RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42,
)


with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:
    random_search.fit(X_train, y_train)

    # Log all hyperparameter combinations
    for i in range(len(random_search.cv_results_["params"])):
        with mlflow.start_run(
            run_name=f"Combination {i + 1}", nested=True
        ) as child_run:
            params = random_search.cv_results_["params"][i]
            mean_test_score = random_search.cv_results_["mean_test_score"][i]

            # Log the parameters and their corresponding score
            mlflow.log_params(params)
            mlflow.log_metric("mean_test_score", mean_test_score)

    # Print the best hyperparameters found by RandomizedSearchCV
    print("Best parameters found: ", random_search.best_params_)

    # 4. Let's first log best parameters that we found above
    mlflow.log_params(random_search.best_params_)

    # Train the model with the best parameters
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)

    # Save the trained model to a file
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Prepare test data
    # X_test = test_processed_data.iloc[:, 0:-1].values
    # y_test = test_processed_data.iloc[:, -1].values

    X_test = test_processed_data.drop(columns=["Potability"])
    y_test = test_processed_data.Potability

    # Load the saved model
    model = pickle.load(open("model.pkl", "rb"))

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate and print performance metrics
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)

    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1_score)

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")

    sign = infer_signature(X_test, random_search.best_estimator_.predict(X_test))
    mlflow.sklearn.log_model(
        random_search.best_estimator_, "Best Model", signature=sign
    )

    mlflow.log_artifact(__file__)

    print("acc", acc)
    print("precision", precision)
    print("recall", recall)
    print("f1-score", f1_score)
