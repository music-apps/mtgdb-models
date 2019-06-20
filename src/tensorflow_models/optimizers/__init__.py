from __future__ import absolute_import, division, print_function

from abc import ABC
import yaml 

import numpy as np
import tensorflow as tf


class BaseOptimizer(ABC):

    def __init__(self, model, config_file):
        super().__init__()

        self.train_step = None
        self.cost = None
        
        try:
            self.config = yaml.load(open(config_file))
        except:
            raise('optimizer: not able to load configuration file')

        self.learning_rate = self.config['optimizer']['learning_rate']
        self.num_classes = self.config['architecture']['num_classes']

        self.defiene_optimizer(model)

    def defiene_optimizer(self, model):
        pass
