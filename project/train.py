import argparse
import os
import collections
from typing import NamedTuple
import logging


import mlflow
import numpy as np
import torch
import torch.multiprocessing

# patch for m1 platform
if torch.backends.quantized.engine is None:
    torch.backends.quantized.engine = 'qnnpack'

import data_loader.data_loaders as module_data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer
from utils import prepare_device

torch.multiprocessing.set_sharing_strategy("file_system")

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config: ConfigParser):
    """
    launch training
    :param config:
    :return:
    """

    # setup data_loader instances
    data_loader = config.init_obj("data_loader", module_data)
    valid_data_loader = data_loader.split_validation()

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch)
    logger = config.get_logger("train")
    logger.info(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config["loss"])
    metrics = [getattr(module_metric, met) for met in config["metrics"]]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj("optimizer", torch.optim, trainable_params)
    lr_scheduler = config.init_obj("lr_scheduler", torch.optim.lr_scheduler, optimizer)

    # mlflow setup
    if config["mlflow"]["experiment_name"]:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
        mlflow.set_experiment(config["mlflow"]["experiment_name"])
        mlflow.log_param("n_gpu", config.config["n_gpu"])
        mlflow.log_param("arch", dict(config.config["arch"]))
        mlflow.log_param("data_loader", dict(config.config["data_loader"]))
        mlflow.log_param("optimizer", config.config["optimizer"])
        mlflow.log_param("loss", config.config["loss"])
        mlflow.log_param("metrics", config.config["metrics"])
        mlflow.log_param("lr_scheduler", config.config["lr_scheduler"])
        mlflow.log_param("trainer", config.config["trainer"])
        mlflow.log_param("mlflow", config.config["mlflow"])

    trainer = Trainer(
        model,
        criterion,
        metrics,
        optimizer,
        config=config,
        device=device,
        data_loader=data_loader,
        valid_data_loader=valid_data_loader,
        lr_scheduler=lr_scheduler,
    )

    trainer.train()


def gcp_value_from_metadata(name: str) -> str:
    """
    get metadata value named name
    :param name:
    :return: a tuple with output and error
    """

    stream = os.popen(
        f"echo $(curl http://metadata.google.internal/computeMetadata/v1/instance/attributes/{name} \
         -H 'Metadata-Flavor: Google')"
    )
    output = stream.read().strip()
    return output


def stop_gcp_instance() -> None:
    """
    stop current gcp instance using metadata
    :return:
    """

    from googleapiclient import discovery
    from googleapiclient.discovery import Resource
    from oauth2client.client import GoogleCredentials

    credentials = GoogleCredentials.get_application_default()
    service: Resource = discovery.build("compute", "v1", credentials=credentials)

    project: str = gcp_value_from_metadata("project_id")
    zone: str = gcp_value_from_metadata("zone")
    instance: str = gcp_value_from_metadata("name")

    if project and zone and instance:
        request = service.instances().stop(
            project=project, zone=zone, instance=instance
        )
        response = request.execute()
        logging.info(f"stop results {response}")

def get_parser():
    parser = argparse.ArgumentParser(description="PyTorch Template")
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        type=str,
        help="config file path (default: None)",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: None)",
    )
    parser.add_argument(
        "-d",
        "--device",
        default=None,
        type=str,
        help="indices of GPUs to enable (default: all)",
    )
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs: NamedTuple = collections.namedtuple("CustomArgs", "flags type target")
    options: list[CustomArgs] = [
        CustomArgs(["--lr", "--learning_rate"], type=float, target="optimizer;args;lr"),
        CustomArgs(
            ["--bs", "--batch_size"], type=int, target="data_loader;args;batch_size"
        ),
    ]
    return parser, options

if __name__ == "__main__":

    config: ConfigParser = ConfigParser.from_args(*get_parser())
    main(config)
    stop_gcp_instance()
