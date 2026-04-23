import mlflow
import pandas as pd

# Set the tracking URI to your DagsHub MLflow instance
mlflow.set_tracking_uri("https://dagshub.com/ankit-gadhwal/mlops_project.mlflow") 

# Specify the model name
model_name = "Superb_model"  # Registered model name

try:
    # Create an MlflowClient to interact with the MLflow server
    client = mlflow.tracking.MlflowClient()

    # Get the latest version of the model in the Production stage
    versions = client.get_latest_versions(model_name, stages=["Production"])

    if versions:
        latest_version = versions[0].version
        run_id = versions[0].run_id  # Fetching the run ID from the latest version
        print(f"Latest version in Production: {latest_version}, Run ID: {run_id}")

        # Construct the logged_model string
        logged_model = f'runs:/{run_id}/{model_name}'
        print("Logged Model:", logged_model)

        # Load the model using the logged_model variable
        loaded_model = mlflow.pyfunc.load_model(logged_model)
        print(f"Model loaded from {logged_model}")

        # Input data for prediction
        data = pd.DataFrame({
            'ph': [3.71608],
            'hardness': [204.89045],
            'tds': [20791.318981],
            'chlorine': [7.300212],
            'sulfate': [368.516441],
            'conductivity': [564.308654],
            'organic_carbon': [10.379783],
            'trihalomethanes': [86.99097],
            'turbidity': [2.963135]
        })

        # Make prediction
        prediction = loaded_model.predict(data)
        print("Prediction:", prediction)
    else:
        print("No model found in the 'Production' stage.")

except Exception as e:
    print(f"Error fetching model: {e}")