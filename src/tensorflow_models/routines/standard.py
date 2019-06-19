
import sys
sys.path.append("..")

import argparse

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import numpy as np

from managers import MananagerMTGDBMetadata
from loaders.singlelabel import YamlToTFRecordLoader
from optimizers.sgdclip import SGDClip
from models.vggish import VGGish
from trainers import Trainer


def standard_routine(args):
    config_file = args.config_file

    session = tf.InteractiveSession()
    # session = tf_debug.LocalCLIDebugWrapperSession(session)

    # a Manager should create Loaders and Trainers according to the config_file
    manager = MananagerMTGDBMetadata(config_file, session=session)

    # define architecture
    model = VGGish(config_file)

    # define optimizer
    optimizer = SGDClip(model, config_file)

    # data sanity checks and conversion into .tfrecords format
    manager.prepare_data()

    # train
    manager.train(model, optimizer)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Standard routine')

    parser.add_argument('config_file')

    standard_routine(parser.parse_args())

    # /home/pablo/exp/tf_example/genre_dortmund/vggish_dourtmouth.yaml