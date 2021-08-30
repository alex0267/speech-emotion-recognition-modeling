
# speech-emotion-recognition-modeling
A tool to create models to detect emotions in audio contents.

### Install

- Git clone the project

- Have a working python 3.9, and install requirements in an activated virtual environment:

If `pip install -r ./requirements.txt` complains about llvm:

`brew install llvm@9`

and: 

`echo 'export PATH="/usr/local/opt/llvm@9/bin:$PATH"' >> ~/.zshrc`
`echo 'export LDFLAGS="-L/usr/local/opt/llvm@9/lib"' >> ~/.zshrc`
`echo 'export CPPFLAGS="-I/usr/local/opt/llvm@9/include"' >> ~/.zshrc`

- Data is stored in GCP buckets and version are tracked with DVC (installed with the requirement file).

You can either authenticate with your personal account (`gcloud auth application-default login`) or set GOOGLE_APPLICATION_CREDENTIALS linked to a authorized service account.

Then, pulling the data is a simple `dvc pull`

- You should then be able to start to train a model with `python train.py -c config.json`. More on this in the README.md file in the project subdirectory.

#### TODO 
- tracking experiments ✅
- versioning data with dvc ✅
   data are stored in the google cloud storage bucket "wewyse-centralesupelec-ftv-data"
- choose a template for DS project
- integrate [this](https://github.com/victoresque/pytorch-template) project template ✅   
- configure dvc backend the first time ✅

      - `dvc init`
      - either use a `gcloud auth application-default login`
      or set an environment variable GOOGLE_APPLICATION_CREDENTIALS
      - then launch this command
      `dvc remote add -d ftv-data gs://wewyse-centralesupelec-ftv-data/raw_data`
      import-url will download data from a distant storage and create dvc file and .gitignore file
       `dvc update ./raw_data.dvc` will bring data 
      reference :
         https://dvc.org/doc/use-cases/sharing-data-and-model-files
         
- build a mvp for the projet ✅
  -  reformat data pre treatment
    - add pitch modulation using pyaudio 
    - add creation of overlapping windows 
  -  add them in a data loader
    -  add the correct model
    -  test a minimal config
    - Normalise tensor at the entrance
    - Add data to mlflow
    - Add upsampling
    - launch with Adam optimizer and SGD with momentum            

- add a tool to help for administrative tasks : 
    - publish a docker image
    - destroy a docker image✅
    
    - launch a gcp instance
    - stop a gcp instance
    - destroy a gcp instance
    
    - launch a training 
    - launch an inference
    
    - launch mlflow server instance
    - stop mlflow server
    - destroy mlflow server instance  

  what's missing : 
    - Todo for use of local tensorboard
    - Document a simple way to launch on GCP
    - Add loss functions
    - More documentation 
    - BUG: when using more than one worker, there's a bug in the serialization process 