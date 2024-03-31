import os
import logging.config
import yaml

from src.utils.constants import LOGGING_CONFIG_FILE
from src.utils.file import load_yml_file


def get_default_logger():
    logger_config = load_yml_file(LOGGING_CONFIG_FILE)
    logging.config.dictConfig(logger_config)       

    return logging.getLogger("default")

LOGGER = get_default_logger()