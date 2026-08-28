import mlflow.data
import pandas as pd
import mlflow.sklearn

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri('http://127.0.0.1:5000')

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest classifier
max_depth = 8

mlflow.autolog()

mlflow.set_experiment('iris_dt')

with mlflow.start_run(run_name='pk_exp_with_confusion_matrix_log_artifact'):

    dt = DecisionTreeClassifier(max_depth=max_depth)
    dt.fit(X_train, y_train)

    # Evaluate the model
    y_pred = dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
 

    # Convert X_train and X_test to pandas DataFrames
    train_df = pd.DataFrame(X_train, columns=iris.feature_names)
    test_df = pd.DataFrame(X_test, columns=iris.feature_names)

    # Add target variable to the DataFrames
    train_df['variety'] = y_train
    test_df['variety'] = y_test
    print('accuracy', accuracy)

    # Convert to MLflow input data format
    train_df_mlflow = mlflow.data.from_pandas(train_df)
    test_df_mlflow = mlflow.data.from_pandas(test_df)


    
    print('accuracy', accuracy)
    # Log confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion matrix')

    # Save the confusion matrix
    plt.savefig("confusion_matrix.png")




    mlflow.set_tag('author', 'sujat')
    mlflow.set_tag('project', 'iris-classification')
    mlflow.set_tag('algorithm', 'decision-tree')