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
from utils.registry_class import ENGINE

from tools import *

if __name__ == '__main__':
    cfg_update = Config(load=True)
    ENGINE.build(dict(type=cfg_update.TASK_TYPE), cfg_update=cfg_update.cfg_dict)
    