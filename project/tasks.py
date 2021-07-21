import os
from pathlib import Path

import googleapiclient.discovery
from invoke import task

DEFAULT_GCP_ZONE = "europe-west1-d"
DEFAULT_PROJECT_NAME = "wewyse-centralesupelec-ftv"


@task(help={'project_name': "project id or name to push to on GCP",
            'local_image_name': "local docker image name ",
            'remote_image_name': "remote docker image name"})
def publish_image(c, project_name=DEFAULT_PROJECT_NAME,
                  local_image_name="speech-emotion-modelling",
                  remote_image_name="ser-modelling"):
    """
    Publish image to google container registry.
    """
    # publish a docker image to GCP
    root = str(Path(__file__).parent.parent)
    dockerfile_path = str(Path(root, "infra", "training", "Dockerfile"))
    os.chdir(root)
    c.run("echo 'passing command `docker build -t {local_image_name}:latest -f {dockerfile_path} .`'")
    c.run(f"docker build -t {local_image_name}:latest -f {dockerfile_path} .")
    c.run("echo 'done'")

    c.run(
        f"echo 'passing command docker tag {local_image_name}:latest eu.gcr.io/{project_name}/{remote_image_name}:latest`'")
    c.run(f"docker tag {local_image_name}:latest eu.gcr.io/{project_name}/{remote_image_name}:latest")
    c.run("echo 'done'")

    c.run(f"echo 'passing command `docker push eu.gcr.io/{project_name}/{remote_image_name}:latest`'")
    c.run(f"docker push eu.gcr.io/{project_name}/{remote_image_name}:latest")
    c.run("echo 'done'")


@task(help={'hostname': "google container registry url",
            'project_id': "project name or id",
            'remote_image_name': "remote docker image name"})
def destroy_image(c, hostname="eu.gcr.io", project_id=DEFAULT_PROJECT_NAME, remote_image_name=None):
    """
    destroy named image in google registry.
    """
    if not remote_image_name:
        c.run("echo 'remote image name parameter is mandatory'")
        return
    # destroy a docker image  in GCP
    c.run(
        f"echo 'passing command `gcloud container images delete eu.gcr.io/wewyse-centralesupelec-ftv/ser-modelling2:latest --force-delete-tags`'")
    c.run(f"gcloud container images delete {hostname}/{project_id}/{remote_image_name}:latest --force-delete-tags")
    c.run("echo 'done'")


@task(help={'project_id': "project name or id",
            'zone': "instance zone"})
def list_instances(c, project_id=DEFAULT_PROJECT_NAME, zone=DEFAULT_GCP_ZONE):
    """
    list instances related to a given project
    """
    compute = googleapiclient.discovery.build('compute', 'v1')
    result = compute.instances().list(project=project_id, zone=zone).execute()
    return result['items'] if 'items' in result else None

@task
def create_instance(name, bucket, project=DEFAULT_PROJECT_NAME, zone=DEFAULT_GCP_ZONE):
    """
    create an instance on GCP
    """
    #source : https://github.com/GoogleCloudPlatform/python-docs-samples/blob/master/compute/api
    compute = googleapiclient.discovery.build('compute', 'v1')
    # Get the latest Debian Jessie image.
    # TODO : to be parameterized
    image_response = compute.images().getFromFamily(project='debian-cloud', family='debian-9').execute()
    source_disk_image = image_response['selfLink']

    # Configure the machine
    # TODO : to be parameterized
    machine_type = f"zones/{zone}/machineTypes/n1-standard-1"

    # TODO : to be parameterized
    startup_script = open(os.path.join(os.path.dirname(__file__), 'startup-script.sh'), 'r').read()
    image_url = "data:image/png;base64,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"
    image_caption = "wewyse"

    config = {
        'name': name,
        'machineType': machine_type,
        # Specify the boot disk and the image to use as a source.
        'disks': [
            {
                'boot': True,
                'autoDelete': True,
                'initializeParams': {
                    'sourceImage': source_disk_image,
                }
            }
        ],
        # Specify a network interface with NAT to access the public
        # internet.
        'networkInterfaces': [{
            'network': 'global/networks/default',
            'accessConfigs': [
                {'type': 'ONE_TO_ONE_NAT', 'name': 'External NAT'}
            ]
        }],
        # Allow the instance to access cloud storage and logging.
        'serviceAccounts': [{
            'email': 'default',
            'scopes': [
                'https://www.googleapis.com/auth/devstorage.read_write',
                'https://www.googleapis.com/auth/logging.write'
            ]
        }],
        # Metadata is readable from the instance and allows you to
        # pass configuration from deployment scripts to instances.
        'metadata': {
            'items': [{
                # Startup script is automatically executed by the
                # instance upon startup.
                'key': 'startup-script',
                'value': startup_script
            }, {
                'key': 'url',
                'value': image_url
            }, {
                'key': 'text',
                'value': image_caption
            }, {
                'key': 'bucket',
                'value': bucket
            }]
        }
    }
    return compute.instances().insert(project=project, zone=zone, body=config).execute()


@task(help={'name': "instance name",
            'project_id': "project name or id",
            'zone': "instance zone"})
def delete_instance(c, name, project_id=DEFAULT_PROJECT_NAME, zone=DEFAULT_GCP_ZONE):
    """
    delete named instance on GCP
    """
    compute = googleapiclient.discovery.build('compute', 'v1')
    return compute.instances().delete(project=project_id, zone=zone, instance=name).execute()