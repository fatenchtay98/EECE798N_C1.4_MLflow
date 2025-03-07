import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from utils import load_config, setup_logging


logger = setup_logging()
config = load_config()

mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])

def preprocess_data():
    """
    Function to preprocess data: handle missing values, encode target, and features
    """
    data = pd.read_csv(config["data"]["filepath"])

    data.replace('?', np.nan, inplace=True)
    data.dropna(inplace=True)

    label_encoder = LabelEncoder()
    data['income'] = label_encoder.fit_transform(data['income'])

    X = data.drop(columns=['income'])
    y = data['income']

    categorical_cols = X.select_dtypes(include=['object']).columns

    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    numerical_cols = ['age', 'fnlwgt', 'capital.gain', 'capital.loss', 'hours.per.week']
    scaler = StandardScaler()
    X[numerical_cols] = scaler.fit_transform(X[numerical_cols])

    return X, y


def log_model_run(model, model_name, params, X_train, X_test, y_train, y_test):
    """
    Function to train the model, log metrics and model artifact into MLflow
    """
    with mlflow.start_run():
        mlflow.log_param("model", model_name)
        for param, value in params.items():
            mlflow.log_param(param, value)

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        input_example = X_train.iloc[[0]].copy()

        mlflow.sklearn.log_model(model, config["model"]["model_directory"], input_example=input_example)

        return accuracy, f1


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train logistic regression with different hyperparameter sets
    """
    lr_params_1 = {"C": 0.1, "solver": "liblinear", "max_iter": 100}
    lr_model_1 = LogisticRegression(C=lr_params_1["C"], solver=lr_params_1["solver"], max_iter=lr_params_1["max_iter"])
    log_model_run(lr_model_1, "Logistic Regression", lr_params_1, X_train, X_test, y_train, y_test)

    # lr_params_2 = {"C": 10.0, "solver": "lbfgs", "max_iter": 200}
    # lr_model_2 = LogisticRegression(C=lr_params_2["C"], solver=lr_params_2["solver"], max_iter=lr_params_2["max_iter"])
    # log_model_run(lr_model_2, "Logistic Regression", lr_params_2, X_train, X_test, y_train, y_test)

    # lr_params_3 = {"C": 0.5, "solver": "saga", "max_iter": 500}
    # lr_model_3 = LogisticRegression(C=lr_params_3["C"], solver=lr_params_3["solver"], max_iter=lr_params_3["max_iter"])
    # log_model_run(lr_model_3, "Logistic Regression", lr_params_3, X_train, X_test, y_train, y_test)

def train_random_forest(X_train, X_test, y_train, y_test):
    """
    Train Random Forest with different hyperparameter sets
    """
    rf_params_1 = {"n_estimators": 50, "max_depth": 10, "min_samples_split": 2}
    rf_model_1 = RandomForestClassifier(n_estimators=rf_params_1["n_estimators"], 
                                       max_depth=rf_params_1["max_depth"], 
                                       min_samples_split=rf_params_1["min_samples_split"])
    log_model_run(rf_model_1, "Random Forest", rf_params_1, X_train, X_test, y_train, y_test)

    # rf_params_2 = {"n_estimators": 100, "max_depth": 15, "min_samples_split": 5}
    # rf_model_2 = RandomForestClassifier(n_estimators=rf_params_2["n_estimators"], 
    #                                    max_depth=rf_params_2["max_depth"], 
    #                                    min_samples_split=rf_params_2["min_samples_split"])
    # log_model_run(rf_model_2, "Random Forest", rf_params_2, X_train, X_test, y_train, y_test)

    # rf_params_3 = {"n_estimators": 200, "max_depth": 20, "min_samples_split": 10}
    # rf_model_3 = RandomForestClassifier(n_estimators=rf_params_3["n_estimators"], 
    #                                    max_depth=rf_params_3["max_depth"], 
    #                                    min_samples_split=rf_params_3["min_samples_split"])
    # log_model_run(rf_model_3, "Random Forest", rf_params_3, X_train, X_test, y_train, y_test)


def run_all_experiments(X_train, X_test, y_train, y_test):
    """
    Run all models and log their metrics
    """
    # train_logistic_regression(X_train, X_test, y_train, y_test)
    train_random_forest(X_train, X_test, y_train, y_test)
