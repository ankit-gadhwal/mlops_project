import pandas as pd
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import pickle
dagshub.init(repo_owner='ankit-gadhwal', repo_name='mlops_project', mlflow=True)
mlflow.set_experiment("water_exp4")
mlflow.set_tracking_uri("https://dagshub.com/ankit-gadhwal/mlops_project.mlflow")
data = pd.read_csv(r"data\Watera.csv")
from sklearn.model_selection import train_test_split,RandomizedSearchCV
train_data,test_data = train_test_split(data,test_size=0.20,random_state=42)
def fill_missing_with_mean(df):
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column].fillna(mean_value,inplace=True)
    return df


# Fill missing values with median
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)


# X_train = train_processed_data.iloc[:,0:-1].values
# y_train = train_processed_data.iloc[:,-1].values

X_train = train_processed_data.drop(columns = ["potability"],axis=1)
y_train = train_processed_data["potability"]
# Define the model and parameter distribution for RandomizedSearchCV
rf = RandomForestClassifier(random_state = 42)
param_dict = {
    'n_estimators': [100,200,300,500,1000],
    'max_depth' : [None,4,6,7,10]
}
  
# perform RandomizedSearchCV to find the best hyperparameters
random_search = RandomizedSearchCV(estimator=rf,param_distributions=param_dict,cv = 5,n_jobs=-1,verbose=2)

with mlflow.start_run(run_name = "Random Forest Tuning") as parent_run:
    random_search.fit(X_train,y_train)
    
    for i in range(len(random_search.cv_results_['params'])):
        with mlflow.start_run(run_name=f"Combination{i+1}",nested=True)as child_run:
            mlflow.log_params(random_search.cv_results_['params'][i])
            mlflow.log_metric("mean_test_score",random_search.cv_results_['mean_test_score'][i])
    # best hyperparameters found by RandomiziedSearchCV
    print("Best parameters found: ",random_search.best_params_)
    mlflow.log_params(random_search.best_params_)
    # train the model with the best parameters
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train,y_train)
    # save 
    pickle.dump(best_rf,open("model.pkl","wb"))

    # prepare test data
    # X_test = test_processed_data.iloc[:,0:-1].values
    # y_test = test_processed_data.iloc[:,-1].values

    X_test = test_processed_data.drop(columns = ["potability"],axis = 1)
    y_test = test_processed_data["potability"]
    # load the saved model
    model = pickle.load(open('model.pkl',"rb"))
    
    y_pred = model.predict(X_test)

    # calculate and print performance metrics
    acc = accuracy_score(y_test,y_pred)
    precision = precision_score(y_test,y_pred)
    recall = recall_score(y_test,y_pred)
    f1_score = f1_score(y_test,y_pred)

    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)

    mlflow.log_input(train_df,"train")
    mlflow.log_input(test_df,"test")
    mlflow.log_metric("accuracy",acc)
    mlflow.log_metric("precision",precision)
    mlflow.log_metric("recall",recall)
    mlflow.log_metric("f1 score",f1_score)
    mlflow.log_artifact(__file__)
    mlflow.sklearn.log_model(random_search.best_estimator_,"best model")
    print("acc",acc)
    print("precision", precision)
    print("recall", recall)
    print("f1-score",f1_score)