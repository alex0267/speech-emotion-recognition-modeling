#### Guidelines for a fresh install and training

---
This guide should permit you to :
- launch symply the training of a deep learning model using pytorch. 
- track your results in a mlflow server

Two cases are considered: 

- you want to train your model and track your results on a distant mlflow server 
    
- you want to train your model and track your results on a local mlflow server 
          

---

- ##### train on an instance and track your experiments in a distant mlflow server
    
    - if you use a distant instance for the training, ssh into it beforehand
    - clone current repository
        ```
         git clone -b pytorch git@github.com:WeWyse/speech-emotion-recognition-modeling.git
        ``` 
    - cd into overall project 
            ```
              cd speech-emotion-recognition-modeling
            ``` 
    - cd into "project" directory
        ```
          cd project
        ``` 
    - install python 3.9
    - activate your python environment 
    - install pip
    - install dependencies
        ```
        pip install -r requirements.txt
        ```
    - install [dvc](https://dvc.org/) 
    - set up dvc using this command 
        ```
        dvc remote add --default ftv-data \
                                         gs://wewyse-centralesupelec-ftv-data/raw_data
        ```

    - grab data locally using the following command 
        ```
        dvc pull
        ```
    - store google service account json file in your home directory  
    - launch the following commands in console
        ```
        export GOOGLE_APPLICATION_CREDENTIALS="<change me>/credentials-mlflow.json"
        export MLFLOW_TRACKING_USERNAME="<change me>"
        export MLFLOW_TRACKING_PASSWORD="<change me>"
        ```
    - check that the mlflow server's instance is running 
    - check that the mlflow service is running 
      if not, launch this command:  
        ```
        mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root gs://wewy
        se-centralesupelec-ftv-mlflow --host localhost
        ```
    - inspect config.json and add the following parameters in the "mlflow" part:
          
             - "experiment_name": "speech-emotion-recognition-modelling" the name of the experiment
             - "tracking_uri": "http://<ip adress of your instance>:80"  uri of your mlflow instance and port 
            
    - launch a training using 
        ```
        python train.py -c config.json
        ``` 
    - You can see your results at  "http://<ip adress of your instance>:5000"
    - You'll be prompted for your login and password. 
      You can change or set a new one using this command in you mlflow instance : 
        ```
        sudo htpasswd -c /etc/nginx/.htpasswd <USERNAME>
        ```
        
- ##### train on an instance and track your experiments in local using mlflow
    Same process but ignore all items related to the distant mlflow server.
    in the config.json file, in the "mlflow" part change the following values
    - "experiment_name": "speech-emotion-recognition-modelling" the name of the experiment
    - "tracking_uri": "./mlruns" local directory to store results and artifacts 
    after training, you can launch a mlflow server in local using :
        ```
        mlflow ui 
        ```

