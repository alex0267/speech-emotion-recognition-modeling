import os
from pathlib import Path

import googleapiclient.discovery
from invoke import task
from invoke import Collection

class Config:
    DEFAULT_GCP_ZONE = "europe-west4-a"
    DEFAULT_PROJECT_NAME = "wewyse-centralesupelec-ftv"
    DEFAULT_REMOTE_IMAGE_NAME = "ser-modelling"
    INSTANCE_DEFAULT_PARAMS = {"project": "debian-cloud", "family": "debian-9"}
    GPU_INSTANCE_DEFAULT_PARAMS = {
        "project": "deeplearning-platform-release",
        "family": "pytorch-latest-gpu",
    }


@task(
    help={
        "project_name": "project id or name to push to on GCP",
        "local_image_name": "local docker image name ",
        "remote_image_name": "remote docker image name",
    }
)
def publish_image(
        c,
        project_name=Config.DEFAULT_PROJECT_NAME,
        local_image_name="speech-emotion-modelling",
        remote_image_name=Config.DEFAULT_REMOTE_IMAGE_NAME,
):
    """
    Publish image to google container registry.
    """
    # publish a docker image to GCP
    root = str(Path(__file__).parent.parent)
    dockerfile_path = str(Path(root, "infra", "training", "Dockerfile"))
    os.chdir(root)
    c.run(
        "echo 'passing command `docker build -t {local_image_name}:latest -f {dockerfile_path} .`'"
    )
    c.run(f"docker build -t {local_image_name}:latest -f {dockerfile_path} .")
    c.run("echo 'done'")

    c.run(
        f"echo 'passing command docker tag {local_image_name}:latest \
         eu.gcr.io/{project_name}/{remote_image_name}:latest`'"
    )
    c.run(
        f"docker tag {local_image_name}:latest eu.gcr.io/{project_name}/{remote_image_name}:latest"
    )
    c.run("echo 'done'")

    c.run(
        f"echo 'passing command `docker push eu.gcr.io/{project_name}/{remote_image_name}:latest`'"
    )
    c.run(f"docker push eu.gcr.io/{project_name}/{remote_image_name}:latest")
    c.run("echo 'done'")


@task(
    help={
        "hostname": "google container registry url",
        "project_id": "project name or id",
        "remote_image_name": "remote docker image name",
    }
)
def destroy_image(
        c,
        hostname="eu.gcr.io",
        project_id=Config.DEFAULT_PROJECT_NAME,
        remote_image_name=None,
):
    """
    destroy named image in google registry.
    """
    if not remote_image_name:
        c.run("echo 'remote image name parameter is mandatory'")
        return
    # destroy a docker image  in GCP
    c.run(
        "echo 'passing command `gcloud container images delete "
        "eu.gcr.io/wewyse-centralesupelec-ftv/ser-modelling2:latest --force-delete-tags`'"
    )
    c.run(
        f"gcloud container images delete {hostname}/{project_id}/{remote_image_name}:latest --force-delete-tags"
    )
    c.run("echo 'done'")


@task(help={"project_id": "project name or id", "zone": "instance zone"})
def list_instances(
        c, project_id=Config.DEFAULT_PROJECT_NAME, zone=Config.DEFAULT_GCP_ZONE
):
    """
    list instances related to a given project
    """
    compute = googleapiclient.discovery.build("compute", "v1")
    result = compute.instances().list(project=project_id, zone=zone).execute()
    return result["items"] if "items" in result else None


@task
def create_instance(
        c,
        name=None,
        remote_image_name=None,
        project_id=Config.DEFAULT_PROJECT_NAME,
        zone=Config.DEFAULT_GCP_ZONE,
        using_gpu=False,
        image_project=None,
        image_family=None,
):
    """
    create an instance on GCP
    using an image stored in google container registry
    invoke create-instance --name=test-instance --remote-image-name=ser-modelling
    using an image stored in google container registry with gpu
    invoke create-instance --name=test-instance --remote-image-name=ser-modelling --using-gpu

    create an instance on GCP
    using a standard image provided by gcp
    invoke create-instance --name=test-instance --image-project=debian-cloud --image-family=debian-9
    ou
    invoke create-instance --name=test-instance --image-project=deeplearning-platform-release
    --image-family=pytorch-latest-cpu

    same command using gpu
    invoke create-instance --name=test-instance --image-project=deeplearning-platform-release
    --image-family=pytorch-latest-gpu --using-gpu

    create an instance on GCP
    using a standard image provided by gcp and using default values
    invoke create-instance --name=test-instance
    same command using gpu
    invoke create-instance --name=test-instance --using-gpu

    """
    if not name:
        c.run("echo 'name parameter is mandatory'")
        return

    metadata_string = f"project_id={project_id},zone={zone},name={name}"
    if using_gpu:
        metadata_string += ",install-nvidia-driver=True"
    if remote_image_name:
        # custom image type
        command = f"gcloud compute instances create-with-container {name}   \
               --boot-disk-size=32G --boot-disk-device-name={name}  \
               --container-image eu.gcr.io/{project_id}/{remote_image_name} --zone={zone} \
               --maintenance-policy=TERMINATE  \
               --scopes https://www.googleapis.com/auth/cloud-platform \
               --metadata='{metadata_string}'"
        if using_gpu:
            command += " --accelerator='type=nvidia-tesla-v100,count=1'"

    else:
        # standard image type
        if image_family and image_project:
            command = f"gcloud compute instances create '{name}' --zone='{zone}'  --maintenance-policy=TERMINATE \
            --image-family='{image_family}' \
            --image-project='{image_project}' \
            --scopes https://www.googleapis.com/auth/cloud-platform \
            --metadata='{metadata_string}'"
            if using_gpu:
                command += " --accelerator='type=nvidia-tesla-v100,count=1'"
        else:
            if using_gpu:
                command = f"gcloud compute instances create '{name}' --zone='{zone}'  --maintenance-policy=TERMINATE \
                --image-family='{Config.GPU_INSTANCE_DEFAULT_PARAMS['family']}' \
                --image-project='{Config.GPU_INSTANCE_DEFAULT_PARAMS['project']}' \
                --accelerator='type=nvidia-tesla-v100,count=1' \
                --scopes https://www.googleapis.com/auth/cloud-platform \
                --metadata='{metadata_string}'"
            else:
                command = f"gcloud compute instances create '{name}' --zone='{zone}'  --maintenance-policy=TERMINATE \
                --image-family='{Config.INSTANCE_DEFAULT_PARAMS['family']}' \
                --image-project='{Config.INSTANCE_DEFAULT_PARAMS['project']}' \
                --scopes https://www.googleapis.com/auth/cloud-platform \
                --metadata='{metadata_string}'"
    c.run(command)


@task(
    help={
        "name": "instance name",
        "project_id": "project name or id",
        "zone": "instance zone",
    }
)
def delete_instance(
        c, name, project_id=Config.DEFAULT_PROJECT_NAME, zone=Config.DEFAULT_GCP_ZONE
):
    """
    delete named instance on GCP
    """
    compute = googleapiclient.discovery.build("compute", "v1")
    return (
        compute.instances()
            .delete(project=project_id, zone=zone, instance=name)
            .execute()
    )


@task(
    help={
        "dirpath": "directory to apply black to",
    }
)
def black(
        c,
        dirpath=".",
):
    """
    apply black to given directory.
    """
    c.run(f"black {dirpath}")


@task(
    help={
        "dirpath": "directory to apply isort",
    }
)
def isort(
        c,
        dirpath=".",
):
    """
    apply isort to given directory.
    """
    c.run(f"isort {dirpath}")


@task(
    help={
        "dirpath": "directory to apply isort",
    }
)
def flake8(
        c,
        dirpath=".",
):
    """
    apply flake8 to given directory.
    """
    c.run(f"flake8 {dirpath}")


@task(
    help={
        "inpath": "directory containing sounds files",
        "outpath": "directory containing pictures files",
        "debug" : "debug mode",
        "limit" : "limit in image transformed"
    }
)
def sound_to_pics(
        c,
        inpath=None,
        outpath=None,
        debug=False,
        limit=None
):
    """
    transform sounds files to melspectrogram from a given dir to another dir
    """
    import warnings
    warnings.filterwarnings('ignore')
    import torch
    # patch for m1 platform
    if torch.backends.quantized.engine is None:
        torch.backends.quantized.engine = 'qnnpack'
    from data_loader.utils import transformations
    transformations(inpath=inpath, outpath=outpath,debug=debug,limit=limit)


ns = Collection()
images = Collection('images')
images.add_task(publish_image, 'publish_image')
images.add_task(destroy_image, 'destroy_image')
ns.add_collection(images)

instances = Collection('instances')
instances.add_task(list_instances, 'list_instances')
instances.add_task(create_instance, 'create_instance')
instances.add_task(delete_instance, 'delete_instance')
ns.add_collection(instances)

maintenance = Collection('maintenance')
maintenance.add_task(black,"black")
maintenance.add_task(flake8,"flake8")
maintenance.add_task(isort,"isort")
ns.add_collection(maintenance)

training = Collection('training')
training.add_task(sound_to_pics,'sound_to_pics')
ns.add_collection(training)