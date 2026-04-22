import mlflow
import dagshub

mlflow.set_tracking_uri("https://dagshub.com/ankit-gadhwal/mlops_project.mlflow")

dagshub.init(repo_owner='ankit-gadhwal', repo_name='mlops_project', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)