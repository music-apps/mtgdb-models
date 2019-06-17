from abc import ABC
import os
import yaml

import numpy as np
from sklearn.preprocessing import LabelBinarizer


class BaseEvaluator(ABC):

    def __init__(self, config_file):
        try:
            self.config = yaml.load(open(config_file))
        except:
            raise('BaseEvaluator: Not able to load configuration file')

        self.filelist = None
        self.groundtruth = None

        self.project_directory = self.config['project_directory']
        self.groundtruth_basefolder = self.config['groundtruth_basefolder']
        self.validation_size = self.config['evaluator']['validation_size']
        self.stratify = self.config['evaluator']['stratify']
        self.test_strategy = self.config['evaluator']['test_strategy']
        
        self.parse()
        self.generate_splits()


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


class Evaluator(BaseEvaluator):

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
        self.matched_filelist = {idx: os.path.join(self.groundtruth_basefolder, '.'.join(self.filelist[key].split('.')[:-1]) + '.npy') for idx, key in zip(self._key_generator(common_n), common_keys)}


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


class EvaluatorFromMTGDBMetadata(Evaluator):

    def _parse_files(self):
        self.filelist = yaml.load(open(self.config['filelist']))
        self.groundtruth = yaml.load(open(self.config['groundtruth']))['groundTruth']
