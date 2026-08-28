from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

import mlflow
import mlflow.data

df = pd.read_csv("https://raw.githubusercontent.com/npradaschnor/Pima-Indians-Diabetes-Dataset/refs/heads/master/diabetes.csv")

# splitting the data into features and target
X= df.drop('Outcome',axis=1)
y = df['Outcome']

# splitting into training and testing sets
X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,random_state=42)

# creating a Random Forest Classifier
rf= RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV

param_grid={
'n_estimators':[10,50,100],
'max_depth':[None,10,20,30]
}
#Applying Grid Search CV

grid_search = GridSearchCV(estimator=rf,param_grid=param_grid,cv=5,n_jobs =- 1,verbose=2)

mlflow.set_experiment('diabetes-rf-hp')

with mlflow.start_run():

    grid_search.fit(X_train, y_train)

    # log all the children
    for i in range(len(grid_search.cv_results_['params'])):
        print(i)
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_['params'][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_['mean_test_score'][i])

    # Displaying the best parameters and best scores

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_  

    #params

    mlflow.log_params(best_params)

    #metrics

    mlflow.log_metric('accuracy',best_score)

    # data
    train_df = X_train.copy()
    train_df['Outcome'] = y_train

    train_df_mlflow = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df_mlflow, "training")

    test_df = X_test.copy()
    test_df['Outcome'] = y_test

    test_df_mlflow = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df_mlflow, "validation")

    #source code
    mlflow.log_artifact(__file__)

    #model
    mlflow.sklearn.log_model(grid_search.best_estimator_,"random forest")

    #tags
    mlflow.set_tag("author","Sujat")

    print(best_params)
    print(best_score)