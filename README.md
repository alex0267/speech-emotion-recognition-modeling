
# speech-emotion-recognition-modeling
A tool to create models to detect emotions in audio contents.


#### TODO 
- tracking experiments ✅
- versioning data with dvc ✅
   data are stored in the google cloud storage bucket "wewyse-centralesupelec-ftv-data"
- choose a template for DS project
- integrate [this](https://github.com/victoresque/pytorch-template) project template ✅   
- configure dvc backend ✅  

      dvc init
      either use a `gcloud auth application-default login``
      or set an environment variable GOOGLE_APPLICATION_CREDENTIALS
      then launch this command
      `dvc remote add -d ftv-data gs://wewyse-centralesupelec-ftv-data/raw_data`
      import-url will download data from a distant storage and create dvc file and .gitignore file
       “dvc update ./raw_data.dvc” will bring data 
      reference :
         https://dvc.org/doc/use-cases/sharing-data-and-model-files
         
- build a mvp for the projet ✅
  -  reformat data pre treatment
    - add pitch modulation
        https://wiki.audacityteam.org/wiki/SoundTouch
        http://www.surina.net/soundtouch/
    - add creation of overlapping windows 
  -  add them in a data loader
    -  add the correct model
    -  test a minimal config
    
  what's missing : 
    - Normalise tensor at the entrance
    - Add upsampling
    - Todo for use of local tensorboard
    - Add data to mlflow
    - Document a simple way to launch on GCP
    - Add loss function
    - More documentation 
    - BUG: when using more than one worker, there's a bug in the serialization process 
