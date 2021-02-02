## Setup mlflow and storage 

####Main steps

Those instructions are meant to be used in the directory below, mlflow-gcp.
This directory includes another README for more details.

```
#activate service account 
gcloud auth activate-service-account mlflow@wewyse-centralesupelec-ftv.iam.gserviceaccount.com --key-file=/Users/ncouturier/.config/gcloud/credentials-mlflow.json
#create cluster 
gcloud container clusters create cluster-mlflow --region europe-west1 --num-nodes=2
#describe cluster
gcloud container clusters describe  cluster-mlflow
# run this if you need to configure your shell to talk to this cluster (e.g. if you restarted your computer)
gcloud container clusters get-credentials cluster-mlflow
#credentials
export GCLOUD_SERVICE_KEY_ENC=$(cat /Users/ncouturier/.config/gcloud/credentials-mlflow.json | base64)
export GCP_STORAGE_BUCKET=speech-emotion-recognition-modeling-mlflow
#populate secrets
source ./populate_secret.sh
#Create Deployment and service
kubectl create -f mlflow-gcp-deployment.yaml
kubectl create -f mlflow-gcp-service.yaml
#get ip adress and port 
kubectl get service
```