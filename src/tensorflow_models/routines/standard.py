
import sys
sys.path.append("..")

import tensorflow as tf
from tensorflow.python import debug as tf_debug

import numpy as np

from evaluators import EvaluatorFromMTGDBMetadata
from loaders.singlelabel import YamlToTFRecordLoader
from optimizers.sgdclip import SGDClip
from models.vggish import VGGish
from trainers import Trainer

def tensorflow_shutup():
    """
    Make Tensorflow less verbose
    """
    try:
        # noinspection PyPackageRequirements
        import os
        from tensorflow import logging
        logging.set_verbosity(logging.ERROR)
        os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

        # Monkey patching deprecation utils to shut it up! Maybe good idea to disable this once after upgrade
        # noinspection PyUnusedLocal
        def deprecated(date, instructions, warn_once=True):
            def deprecated_wrapper(func):

                return func
            return deprecated_wrapper

        from tensorflow.python.util import deprecation
        deprecation.deprecated = deprecated

    except ImportError:
        pass

tensorflow_shutup()

session = tf.InteractiveSession()
# session = tf_debug.LocalCLIDebugWrapperSession(session)

config_file = '/home/pablo/exp/tf_example/genre_dortmund/vggish_dourtmouth.yaml'

evaluator = EvaluatorFromMTGDBMetadata(config_file)


# Load data and labels listed in filelist and groundtruth
#and makes sure that the data is correctly stores in tfrecords files
train_loader = YamlToTFRecordLoader('/home/pablo/exp/tf_example/genre_dortmund/train_filelist.yaml',
                                    '/home/pablo/exp/tf_example/genre_dortmund/train_groundtruth.yaml',
                                    '/home/pablo/exp/tf_example/genre_dortmund/data/',
                                    'train',
                                    config_file,
                                    evaluator.label_binarizer,
                                    session=session)

val_loader = YamlToTFRecordLoader('/home/pablo/exp/tf_example/genre_dortmund/val_filelist.yaml',
                                  '/home/pablo/exp/tf_example/genre_dortmund/val_groundtruth.yaml',
                                  '/home/pablo/exp/tf_example/genre_dortmund/data/',
                                  'val',
                                  config_file,
                                  evaluator.label_binarizer,
                                  session=session)

# define architecture
model = VGGish(config_file)

# define optimizer
optimizer = SGDClip(model, config_file)

# define trainer
trainer = Trainer(model,train_loader, optimizer, config_file, session=session, val_loader=val_loader)

trainer.train()