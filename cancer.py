import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import lazypredict
from lazypredict.Supervised import LazyRegressor
import mlflow
import mlflow.sklearn

# Load the breast cancer dataset
cancer = datasets.load_breast_cancer()
X = pd.DataFrame(cancer.data, columns=cancer.feature_names)
y = pd.Series(cancer.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():

    # Create LazyRegressor model
    reg = LazyRegressor(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = reg.fit(X_train, X_test, y_train, y_test)

    # Log the models and their mean squared errors to MLflow
    for model_name, model in models.items():
        with mlflow.start_run(nested=True):
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions and calculate mean squared error
            y_pred = model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)

            # Log model and mean squared error to MLflow
            mlflow.sklearn.log_model(model, model_name)
            mlflow.log_metric("mean_squared_error", mse)
