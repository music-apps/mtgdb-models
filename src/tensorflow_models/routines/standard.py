
import os
import sys
import argparse
import yaml

import tensorflow as tf
#from tensorflow.python import debug as tf_debug
import numpy as np

from essentia.pytools.extractors import batch_melbands_extractor

sys.path.append("..")
from managers import MananagerMTGDBMetadata
from loaders.singlelabel import YamlToTFRecordLoader
from optimizers.adam import Adam
from models.vggish import VGGish
from trainers import Trainer


def standard_routine(args):
    config_file = args.config_file
    recompute_features = args.recompute_features

    try:
        config = yaml.load(open(config_file))
    except:
        raise('Not able to load configuration file')

    # Project paths
    project_directory = config['project_directory']
    datasets_directory = config['datasets_directory']
    raw_data_directory = config['raw_data_directory']
    results_directory = config['results_directory']

    os.makedirs(project_directory, exist_ok=True)
    os.makedirs(datasets_directory, exist_ok=True)
    os.makedirs(raw_data_directory, exist_ok=True)
    os.makedirs(results_directory, exist_ok=True)
    
    # Input audio dir
    audio_directory = config['audio_directory']
    assert os.path.exists(audio_directory), '{} folder does not exist'.format(audio_directory)

    # Feature extraction
    batch_melbands_extractor(audio_directory, raw_data_directory,
                             generate_log=True,
                             skip_analyzed=not recompute_features,
                             frame_size=config['features']['frame_size'],
                             hop_size=config['features']['hop_size'],
                             number_bands=config['features']['number_bands'],
                             sample_rate=config['features']['sample_rate'],
                             max_frequency=config['features']['max_frequency'],
                             compression_type=config['features']['compression_type'])
                             # normalize=config['features']['normalize'])

    session = tf.InteractiveSession()
    # session = tf_debug.LocalCLIDebugWrapperSession(session)

    # a Manager should create Loaders and Trainers according to the config_file
    manager = MananagerMTGDBMetadata(config_file, session=session)

    # define architecture
    model = VGGish(config_file)

    # define optimizer
    optimizer = Adam(model, config_file)

    # data sanity checks and conversion into .tfrecords format
    manager.prepare_data()

    # train
    manager.train(model, optimizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Standard routine')

    parser.add_argument('config_file')
    parser.add_argument('--recompute_features', '-rf', action='store_true')

    standard_routine(parser.parse_args())
