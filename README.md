
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
         
- build a mvp for the projet
   -  reformat data pre treatment
   -  add them in a data loader
   -  add the correct model
   -  test a minimal config
