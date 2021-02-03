import os
import mlflow
from mlflow import set_tracking_uri, set_experiment
from pathlib import Path

# Set path to service account json file
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/ncouturier/.config/gcloud/credentials-mlflow.json"
os.environ['MLFLOW_TRACKING_USERNAME'] = "<change me>"
os.environ['MLFLOW_TRACKING_PASSWORD'] = "<change me>"

set_tracking_uri("http://34.76.125.135:80")
set_experiment("speech-emotion-recognition-modelling")


# Log a parameter (key-value pair)
mlflow.log_param("param1", 42)

# Log a metric; metrics can be updated throughout the run
mlflow.log_metric("foo", 42 + 1)
mlflow.log_metric("foo", 42 + 2)
mlflow.log_metric("foo", 42 + 3)
#log artifacts
mlflow.log_artifacts(str(Path(Path(__file__).parent,"README.md")))