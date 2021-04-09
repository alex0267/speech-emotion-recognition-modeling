
This details the expected working to dump locally recordings saved in GCP Buckets
from the Speech Web App and convert them into wav files.

1. Install the required dependencies from the `requirements-dump.txt`
2. Install ffmpeg with `brew install ffmpeg lame`
3. In a shell, place yourself in the `utils` directory
4. Run: `python3 ./dump_data_from_gcs.py`

This should lead to the download, conversion to wav and transfer into the `raw_data/french` subfolder.

NB: By default, the files will be added to dvc tracking, comment / remove the following lines if it is
not the desired behavior:

```python
            subprocess.run(
                ["dvc", "add", final_file],
                stdout=subprocess.PIPE,
                stdin=subprocess.PIPE,
            )

```

NB2: Please refer to Speech Web App to learn how to handle the configuration and the required environmental
variable together with Google Cloud authentification.