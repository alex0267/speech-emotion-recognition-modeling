import logging
import os
from pathlib import Path

from invoke import task


@task(help={'project_name': "project id or name to push to on GCP",
            'local_image_name': "local docker image name ",
            'remote_image_name': "remote docker image name"})
def publish_image(c, project_name="wewyse-centralesupelec-ftv",
                  local_image_name="speech-emotion-modelling",
                  remote_image_name="ser-modelling"):
    # publish a docker image to GCP
    root = str(Path(__file__).parent.parent)
    dockerfile_path = str(Path(root, "infra", "training", "Dockerfile"))
    os.chdir(root)
    logging.info(f"passing command `docker build -t {local_image_name}:latest -f {dockerfile_path} .`")
    c.run(f"docker build -t {local_image_name}:latest -f {dockerfile_path} .")
    c.run("echo 'done'")

    logging.info(
        f"passing command docker tag {local_image_name}:latest eu.gcr.io/{project_name}/{remote_image_name}:latest`")
    c.run(f"docker tag {local_image_name}:latest eu.gcr.io/{project_name}/{remote_image_name}:latest")
    c.run("echo 'done'")

    logging.info(f"passing command `docker push eu.gcr.io/{project_name}/{remote_image_name}:latest`")
    c.run(f"docker push eu.gcr.io/{project_name}/{remote_image_name}:latest")
    c.run("echo 'done'")


@task
def destroy_image(c, hostname="eu.gcr.io", project_id="wewyse-centralesupelec-ftv", remote_image_name=None):
    if not remote_image_name:
        c.run("echo 'remote image name parameter is mandatory'")
        return
    # destroy a docker image  in GCP
    logging.info(
        "passing command `gcloud container images delete eu.gcr.io/wewyse-centralesupelec-ftv/ser-modelling2:latest --force-delete-tags`")
    c.run(f"gcloud container images delete {hostname}/{project_id}/{remote_image_name}:latest --force-delete-tags")
    c.run("echo 'done'")
