import argparse

import dill
import torch
import torch.multiprocessing

# patch for m1 platform
if torch.backends.quantized.engine is None:
    torch.backends.quantized.engine = 'qnnpack'

from tqdm import tqdm

import data_loader.data_loaders as module_data
import dataset.datasets as module_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser

torch.multiprocessing.set_sharing_strategy("file_system")


def main(config):
    logger = config.get_logger("test")

    # setup dataset instance
    dataset = config.init_obj("dataset", module_dataset)

    data_loader = config.init_obj("data_loader", module_data,dataset=dataset)
    # setup data_loader instances
    data_loader = getattr(module_data, config["data_loader"]["type"])(
        config["data_loader"]["args"]["data_dir"],
        batch_size=24,
        validation_split=0.0,
        num_workers=config["data_loader"]["args"]["num_workers"],
        dataset=dataset)

    # build model architecture
    model = config.init_obj("arch", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, config["loss"])
    metric_fns = [getattr(module_metric, met) for met in config["metrics"]]

    logger.info("Loading checkpoint: {} ...".format(config.resume))
    checkpoint = torch.load(config.resume, pickle_module=dill)
    state_dict = checkpoint["state_dict"]
    if config["n_gpu"] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    with torch.no_grad():
        for i, (data, target) in enumerate(tqdm(data_loader)):

            data, target = data.to(device), target.to(device)
            output = model(data)

            #
            # save sample images, or do something with output here
            #

            # computing loss, metrics on test set
            loss = loss_fn(output, target)
            batch_size = data.shape[0]
            total_loss += loss.item() * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output, target) * batch_size

    n_samples = len(data_loader.sampler)
    log = {"loss": total_loss / n_samples}
    log.update(
        {
            met.__name__: total_metrics[i].item() / n_samples
            for i, met in enumerate(metric_fns)
        }
    )
    logger.info(log)


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
    return parser


if __name__ == "__main__":
    config = ConfigParser.from_args(*get_parser())
    main(config)
