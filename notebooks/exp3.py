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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
dagshub.init(repo_owner='ankit-gadhwal', repo_name='mlops_project', mlflow=True)
mlflow.set_experiment("water_exp1")
mlflow.set_tracking_uri("https://dagshub.com/ankit-gadhwal/mlops_project.mlflow")
data = pd.read_csv(r"data\Watera.csv")
from sklearn.model_selection import train_test_split
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

from sklearn.ensemble import RandomForestClassifier
import pickle
X_train = train_processed_data.drop(columns=["potability"], axis=1)
y_train = train_processed_data["potability"]

X_test = test_processed_data.drop(columns=["potability"], axis=1)
y_test = test_processed_data["potability"]
models = {
    "Logistic Regression" : LogisticRegression(),
    "Random Forest" : RandomForestClassifier(),
    "Support vector machine" : SVC(),
    "KNeighborsClassifier" : KNeighborsClassifier(),
    "xgboost" : XGBClassifier(),
    "DecisionTreeClassifier" : DecisionTreeClassifier()
}

with mlflow.start_run(run_name = "Water Potability Models Experiments"):
    for model_name,model in models.items():
        # start a child run within the parent run for each individual model
        with mlflow.start_run(run_name = model_name,nested=True):
            # train the model on the training data
            model.fit(X_train,y_train)
            
            
    # save 
            model_filename = f"{model_name.replace(' ','_')}.pkl"
            pickle.dump(model,open("model_filename","wb"))
        # make prediction
            y_pred = model.predict(X_test)

            acc = accuracy_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            f1 = f1_score(y_test,y_pred)

            mlflow.log_metric("acc",acc)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1-score",f1)
    
            cm = confusion_matrix(y_test,y_pred)
            plt.figure(figsize = (5,5))
            sns.heatmap(cm,annot = True)
            plt.xlabel("predicted")
            plt.ylabel("Actual")
            plt.title("confusion matric")
            plt.savefig("confusion_matrix.png")
            mlflow.log_artifact("confusion_matrix.png")
    # log the model to Mlflow
            mlflow.sklearn.log_model(model,model_name.replace(' ','_'))
    
    # ḷog the source code file for reproducibility (the current script)
            mlflow.log_artifact(__file__)

            mlflow.set_tag("author","Ankit Gadhwal")
    print("All models have been trained and logged as child runs successfully.")
    