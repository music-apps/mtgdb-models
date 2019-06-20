from __future__ import absolute_import, division, print_function

from abc import ABC
import yaml

import numpy as np
import tensorflow as tf


class BaseModel(ABC):

    def __init__(self, config_file):
        super().__init__()

        try:
            self.config = yaml.load(open(config_file))
        except:
            raise('model: not able to load configuration file')

        self.num_frames = self.config['architecture']['num_frames']
        self.num_bands = self.config['architecture']['num_bands']
        self.num_classes = self.config['architecture']['num_classes']
        self.input_op_name = self.config['architecture']['input_op_name']
        self.output_op_name = self.config['architecture']['output_op_name']

        #( batch, timestamps, number of bands, channel)
        self.x_shape = [None, self.num_frames, self.num_bands, 1]
        self.y_shape = [None, self.num_classes]

        self.input = tf.placeholder(tf.float32, shape=self.x_shape, name=self.input_op_name)
        self.is_training = tf.placeholder(tf.bool, shape= None, name='is_training')

        self.define_model()


    def define_model(self):
        pass
