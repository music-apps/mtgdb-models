import sys
sys.path.append("..")

from abc import ABC
import os
import yaml

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer

from loaders.singlelabel import YamlToTFRecordLoader
from trainers import Trainer

class BaseManager(ABC):

    def __init__(self, config_file, session=None):
        try:
            self.config_file = config_file
            self.config = yaml.load(open(config_file))
        except:
            raise('BaseManager: Not able to load configuration file')

        self.filelist = None
        self.groundtruth = None

        self.project_directory = self.config['project_directory']
        self.filelist_basefolder = self.config['filelist_basefolder']
        self.datasets_directory = self.config['datasets_directory']
        self.validation_size = self.config['evaluator']['validation_size']
        self.stratify = self.config['evaluator']['stratify']
        self.test_strategy = self.config['evaluator']['test_strategy']

        if not session:
            self.session = tf.InteractiveSession()
            self.owns_session = True
        else:
            self.session = session
            self.owns_session = False
        
        self.parse()


    @staticmethod
    def _key_generator(l):
        # Get the number of digits
        width = 1 + int(np.floor(np.log(float(l-1)) / np.log(10.)))
        
        for i in range(l):
            yield '{:0{width}}'.format(i, width=width)

    def parse(self):
        pass

    def generate_splits(self):
        pass


class Manager(BaseManager):

    def _parse_files(self):
        self.filelist = yaml.load(open(self.config['filelist']))
        self.groundtruth = yaml.load(open(self.config['groundtruth']))

    def parse(self):
        self._parse_files()

        filelist_keys = self.filelist.keys()
        groundtruth_keys = self.groundtruth.keys()

        labels = list(self.groundtruth.values())
        self.label_binarizer = LabelBinarizer()
        self.label_binarizer.fit(labels)

        common_keys = set(filelist_keys).intersection(set(groundtruth_keys))
        common_n = len(common_keys)
        print('Evaluator: Groundtruth contains labels for {}/{} ids'.format(common_n,
                                                                         len(filelist_keys)))

        self.matched_groundtruth = {idx: self.groundtruth[key] for idx, key in zip(self._key_generator(common_n), common_keys)}
        self.matched_filelist = {idx: os.path.join(self.filelist_basefolder, self.filelist[key]) + '.npy' for idx, key in zip(self._key_generator(common_n), common_keys)}


    def generate_splits(self):
        statregy = self.test_strategy['type']
        
        if statregy == 'single_fold':
            self.generate_single_fold_splits()
        
        elif statregy == 'crossfold':
            raise 'crossfold not implemente'
        
        else:
            raise 'Evaluator: test strategy not implemented'


    def generate_single_fold_splits(self):
        from sklearn.model_selection import train_test_split
        
        test_size = self.test_strategy['test_size']
        validation_size = self.validation_size

        ids = list(self.matched_groundtruth.keys())
        labels = list(self.matched_groundtruth.values())

        ids_train, ids_test, labels_train, _ = train_test_split(ids, labels,
                                                                test_size=test_size,
                                                                stratify=labels)

        # Correct the percentage givein that the test data is already split
        corrected_validation_size = validation_size * len(ids) / float(len(ids_train))

        ids_train, ids_val = train_test_split(ids_train,
                                              test_size=corrected_validation_size,
                                              stratify=labels_train)

        ids_dict = {
                'train': ids_train,
                'val': ids_val,
                'test': ids_test
            }

        for name, ids in ids_dict.items():
            groundtruth_filename = os.path.join(
                self.project_directory, '{}_groundtruth.yaml'.format(name))

            filelist_filename = os.path.join(
                self.project_directory, '{}_filelist.yaml'.format(name))
            groundtruth = {str(k): v for k, v
                           in self.matched_groundtruth.items() if k in ids}

            filelist = {str(k): v for k, v 
                        in self.matched_filelist.items() if k in ids}

            with open(groundtruth_filename, 'w') as f:
                yaml.dump(groundtruth, f)

            with open(filelist_filename, 'w') as f:
                yaml.dump(filelist, f)

        self.train_groundtruth_file = os.path.join(self.project_directory, 'train_groundtruth.yaml')
        self.train_filelist_file = os.path.join(self.project_directory, 'train_filelist.yaml')
        self.val_groundtruth_file = os.path.join(self.project_directory, 'val_groundtruth.yaml')
        self.val_filelist_file = os.path.join(self.project_directory, 'val_filelist.yaml')
        self.test_groundtruth_file = os.path.join(self.project_directory, 'test_groundtruth.yaml')
        self.test_filelist_file = os.path.join(self.project_directory, 'test_filelist.yaml')

    def prepare_data(self):
        self.generate_splits()

        # Load data and labels listed in filelist and groundtruth
        #and makes sure that the data is correctly stores in tfrecords files
        self.train_loader = YamlToTFRecordLoader(self.train_filelist_file,
                                                 self.train_groundtruth_file,
                                                 self.datasets_directory,
                                                 'train',
                                                 self.config_file,
                                                 self.label_binarizer,
                                                 session=self.session)

        self.val_loader = YamlToTFRecordLoader(self.val_filelist_file,
                                               self.val_groundtruth_file,
                                               self.datasets_directory,
                                               'val',
                                               self.config_file,
                                               self.label_binarizer,
                                               session=self.session)

        self.test_loader = YamlToTFRecordLoader(self.test_filelist_file,
                                                self.test_groundtruth_file,
                                                self.datasets_directory,
                                                'test',
                                                self.config_file,
                                                self.label_binarizer,
                                                session=self.session)


    def train(self, model, optimizer):

        self.trainer = Trainer(model,
                               self.train_loader, optimizer,
                               self.config_file,
                               val_loader=self.val_loader,
                               session=self.session)

        self.trainer.train()


class MananagerMTGDBMetadata(Manager):

    def _parse_files(self):
        self.filelist = yaml.load(open(self.config['filelist']))
        self.groundtruth = yaml.load(open(self.config['groundtruth']))['groundTruth']
