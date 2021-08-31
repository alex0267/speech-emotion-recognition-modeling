import os

PRIVATE_KEY = os.environ.get("CREDENTIALS_PRIVATE_KEY")[
    :-1
]  # removing trailing line break that breaks generated json file
PRIVATE_KEY_ID = os.environ.get("CREDENTIALS_PRIVATE_KEY_ID")
CLIENT_ID = os.environ.get("CREDENTIALS_CLIENT_ID")

file_content = f"""{{
  "type": "service_account",
  "project_id": "wewyse-centralesupelec-ftv",
  "private_key_id": "{PRIVATE_KEY_ID}",
  "private_key": "{PRIVATE_KEY}",
  "client_email": "terraform-307@wewyse-centralesupelec-ftv.iam.gserviceaccount.com",
  "client_id": "{CLIENT_ID}",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/\
  terraform-307%40wewyse-centralesupelec-ftv.iam.gserviceaccount.com"
}}"""

with open("credentials.json", "w") as f:
    f.write(file_content)
