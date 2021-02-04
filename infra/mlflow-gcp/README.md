## MLFLOW on an instance using sqlite as a database 

## Reference : 
https://towardsdatascience.com/managing-your-machine-learning-experiments-with-mlflow-1cd6ee21996e
##Steps 

#### Activate your service account 
```
gcloud auth activate-service-account mlflow@wewyse-centralesupelec-ftv.iam.gserviceaccount.com --key-file=/Users/ncouturier/.config/gcloud/credentials-mlflow.json
```

#### Create a compute Engine VM instance

```
gcloud compute instances create mlflow-server \
--machine-type n1-standard-1 \
--zone europe-west1-b \
--tags mlflow-server \
--metadata startup-script='#! /bin/bash
sudo apt update
sudo apt-get -y install tmux
echo Installing python3-pip
sudo apt install -y python3-pip
export PATH="$HOME/.local/bin:$PATH"
echo Installing mlflow and google_cloud_storage
pip3 install mlflow google-cloud-storage'
```

#### Firewall rules
```
gcloud compute firewall-rules create mlflow-server \
--direction=INGRESS --priority=999 --network=default \
--action=ALLOW --rules=tcp:5000 --source-ranges=0.0.0.0/0 \
--target-tags=mlflow-server
```

#### bucket for artifacts
```
gsutil mb -l EUROPE-WEST1 gs://wewyse-centralesupelec-ftv-mlflow 
```

#### internal ip /  external ip
10.132.0.56 / 34.76.125.135

#### intermediate commands 

```
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root gs://wewyse-centralesupelec-ftv-mlflow \
--host 10.132.0.56
```

```
mlflow server \
--backend-store-uri sqlite:///mlflow.db \
--default-artifact-root gs://wewyse-centralesupelec-ftv-mlflow \
--host localhost
```


#### add enough rights in this case for this project
 ###### Rights can be set per project
 ```
 gcloud projects add-iam-policy-binding wewyse-centralesupelec-ftv --role roles/editor  --member serviceAccount:mlflow@wewyse-centralesupelec-ftv.iam.gserviceaccount.com
 ```
 ###### Rights can be set in general 
```
gcloud iam service-accounts add-iam-policy-binding mlflow@wewyse-centralesupelec-ftv.iam.gs
erviceaccount.com --member='allAuthenticatedUsers'  --role='roles/owner'   
```
Beware, some rights can't be set in general, some rights can't be set per project
  

