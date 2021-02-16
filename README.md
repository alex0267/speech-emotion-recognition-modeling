
# speech-emotion-recognition-modeling
A tool to create models to detect emotions in audio contents.


#### TODO 
- tracking experiments ✅
- versioning data with dvc ✅
   data are stored in the google cloud storage bucket "wewyse-centralesupelec-ftv-data"
- choose a template for DS project 
   intègre le template ✅  
   configure dvc backend ✅  
      dvc init
      either use a `gcloud auth application-default login``
      or set an environment variable GOOGLE_APPLICATION_CREDENTIALS
      then launch this command
      `dvc remote add -d ftv-data gs://wewyse-centralesupelec-ftv-data/raw_data`
      import-url will download data from a distant storage and create dvc file and .gitignore file
      reference :
         https://dvc.org/doc/use-cases/sharing-data-and-model-files
   faire une chaine à partir des exemples
      