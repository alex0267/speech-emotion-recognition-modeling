import logging
import logging.config
from logging import handlers
from pathlib import Path
from time import sleep

from utils import read_json


def setup_logging(
    save_dir, log_config="logger/logger_config.json", default_level=logging.INFO
):
    """
    Setup logging configuration
    """
    log_config = Path(log_config)
    if log_config.is_file():
        config = read_json(log_config)
        # modify logging paths based on run config
        for _, handler in config["handlers"].items():
            if "filename" in handler:
                handler["filename"] = str(save_dir / handler["filename"])

        logging.config.dictConfig(config)
    else:
        print(
            "Warning: logging configuration file is not found in {}.".format(log_config)
        )
        logging.basicConfig(level=default_level)


# from: https://fanchenbao.medium.com/python3-logging-with-multiprocessing-f51f460b8778
def listener_process(
    queue, save_dir, log_config="logger/logger_config.json", default_level=logging.INFO
):
    setup_logging(save_dir, log_config, default_level)
    while True:
        while not queue.empty():
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
        sleep(1)


def logging_buffer(queue):
    h = handlers.QueueHandler(queue)
    root = logging.getLogger()
    root.addHandler(h)
    root.setLevel(logging.DEBUG)
