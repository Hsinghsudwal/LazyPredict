import pandas as pd
from sklearn import datasets
import lazypredict
from lazypredict.Supervised import LazyClassifier
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load Iris dataset
iris = datasets.load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = pd.Series(iris.target)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow run
with mlflow.start_run():

    # Create LazyClassifier model
    clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
    models, predictions = clf.fit(X_train, X_test, y_train, y_test)

    # Log the models and their accuracies to MLflow
    for model_name, model in models.items():
        with mlflow.start_run(nested=True):
            # Train the model
            model.fit(X_train, y_train)

            # Make predictions and calculate accuracy
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            # Log model and accuracy to MLflow
            mlflow.sklearn.log_model(model, model_name)
            mlflow.log_metric("accuracy", accuracy)
