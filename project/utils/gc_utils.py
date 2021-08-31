from google.api_core.exceptions import BadRequest
from google.cloud import storage

from .config import config, CONFIG_ENV


# get client
def get_gcs_client():
    return storage.Client.from_service_account_json(
        config[CONFIG_ENV].GOOGLE_APPLICATION_CREDENTIALS_PATH
    )


# Checking if a bucket exists and if you have access to it


def check_access(bucket_name):
    try:
        get_gcs_client().get_bucket(bucket_name)
    except BadRequest as exception:
        raise ValueError(
            f"Unable to connect to bucket={bucket_name!r}, "
            f"because bucket not found due to {exception}"
        )
    else:
        print("It exists and we have access to it.")


# file exist
def key_existing(client, bucket_name, key):
    """return a tuple of (
        key's size if it exists or 0,
        S3 key metadata
    )
    If the object doesn't exist, return None for the metadata.
    """
    bucket = client.get_bucket(bucket_name)
    blob = bucket.get_blob(key)
    if blob:
        return blob.size, blob.metadata
    return 0, None


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    # bucket_name = "your-bucket-name"
    # source_file_name = "local/path/to/file"
    # destination_blob_name = "storage-object-name"

    storage_client = get_gcs_client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print("File {} uploaded to {}.".format(source_file_name, destination_blob_name))


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    # bucket_name = "your-bucket-name"
    # source_blob_name = "storage-object-name"
    # destination_file_name = "local/path/to/file"

    storage_client = get_gcs_client()

    bucket = storage_client.bucket(bucket_name)

    # Construct a client side representation of a blob.
    # Note `Bucket.blob` differs from `Bucket.get_blob` as it doesn't retrieve
    # any content from Google Cloud Storage. As we don't need additional data,
    # using `Bucket.blob` is preferred here.
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)

    print("Blob {} downloaded to {}.".format(source_blob_name, destination_file_name))
