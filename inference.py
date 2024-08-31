import os
import sys
import copy
import json
import math
import random
import logging
import itertools
import numpy as np

from utils.config import Config
from utils.registry_class import INFER_ENGINE

from tools import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

if __name__ == '__main__':
    logger.info("Loading configuration")
    cfg_update = Config(load=True)
    logger.info("Building inference engine")
    INFER_ENGINE.build(dict(type=cfg_update.TASK_TYPE), cfg_update=cfg_update.cfg_dict)
    logger.info("Inference engine built successfully")

