import os
import yaml
from subprocess import call

from argparse import ArgumentParser
from path_config import DATASETS_DATA, DATASETS_GROUNDTRUTH, MTGDB_DIR, MUSICNN_DIR

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


def audio_to_repr_path(audio_path, dataset):
    audio_path = '.'.join(audio_path.split('.')[:-1]) + '.pk'
    return 'audio_representation/{}__time-freq/{}'.format(dataset, audio_path)


def extract_features(data_dir, exp_dir, profile=None,
                     skip_analyzed=True, store_frames=False, format='yaml'):
    for dataset, data_paths in DATASETS_DATA.items():
        print('Processing "{}"'.format(dataset))

        dataset_out_dir = os.path.join(data_dir, dataset)
        exp_out_dir = os.path.join(exp_dir, dataset)

        index_file = os.path.join(dataset_out_dir, 'index.tsv')
        gt_train = os.path.join(dataset_out_dir, 'gt_train.csv')
        gt_val = os.path.join(dataset_out_dir, 'gt_val.csv')

        os.makedirs(dataset_out_dir, exist_ok=True)
        os.makedirs(exp_out_dir, exist_ok=True)

        for audio_path in data_paths:
            dataset_in_dir = os.path.join(MTGDB_DIR, 'stable', audio_path)

            groundtruth_input_file = os.path.join(
                MTGDB_DIR, 'stable', DATASETS_GROUNDTRUTH[dataset])

            groundtruth = yaml.load(open(groundtruth_input_file))

            paths = groundtruth['groundTruth'].keys()
            labels = groundtruth['groundTruth'].values()
            ids = list(range(len(labels)))

            genres = np.array(list(set(groundtruth['groundTruth'].values())))
            genres = genres.reshape(-1, 1)

            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(genres)

            labels = np.array(list(labels)).reshape(-1, 1)
            gt_ohe = list(enc.transform(labels).toarray())

            ids_train, ids_val = train_test_split(ids, stratify=gt_ohe)

            gt_train_list = ['ID{:04}\t{}'.format(idx, list(gt_ohe[idx])) for idx in ids_train]
            gt_val_list = ['ID{:04}\t{}'.format(idx, list(gt_ohe[idx])) for idx in ids_val]
            audio_paths = ['ID{:04}\t{}'.format(idx, audio_to_repr_path(path, dataset))
                           for idx, path in zip(ids, paths)]

            with open(index_file, 'w') as f:
                f.write('\n'.join(audio_paths))

            with open(gt_train, 'w') as f:
                f.write('\n'.join(gt_train_list))

            with open(gt_val, 'w') as f:
                f.write('\n'.join(gt_val_list))

            config_file_template = open(os.path.join(MUSICNN_DIR, 'src', 'config_file_template.yaml')).read()

            config_file = os.path.join(MUSICNN_DIR, 'src', 'config_file.yaml')
            config_file_log = os.path.join(exp_out_dir, 'config_file.yaml')

            configured_file = config_file_template % {'dataset': dataset,
                                                      'data_folder': os.path.abspath(dataset_out_dir) + '/',
                                                      'audio_folder': os.path.abspath(dataset_in_dir) + '/',
                                                      'identifier': dataset,
                                                      'index_file': 'index.tsv',
                                                      'gt_train': gt_train,
                                                      'gt_val': gt_val,
                                                      'num_classes_dataset': len(genres)
                                                      }

            # write the project file
            with open(config_file_log, 'w') as pfile:
                pfile.write(configured_file)

            # write the project file
            with open(config_file, 'w') as pfile:
                pfile.write(configured_file)

            # Compute feaures
            # script = os.path.join(MUSICNN_DIR, 'src', 'preprocess_librosa.py')
            # call(['python', script, 'mtgdb_spec'], cwd=os.path.dirname(script))

            # Train model
            script = os.path.join(MUSICNN_DIR, 'src', 'train.py')
            call(['python', script, 'spec'], cwd=os.path.dirname(script))
            # call(['CUDA_VISIBLE_DEVICES=0', 'python', script, 'spec'], cwd=os.path.dirname(script))

            # Evaluate model
            # script = os.path.join(MUSICNN_DIR, 'src', 'train.py')
            # call(['CUDA_VISIBLE_DEVICES=0', 'python', script, 'spec'], cwd=os.path.dirname(script))


if __name__ == '__main__':
    argumentParser = ArgumentParser(
        'Generate music_extractor features for all the MTGDB datasets relevant to train Gaia models.')
    argumentParser.add_argument('data_dir',
                                help='Where to store the .npy files.')
    argumentParser.add_argument('exp_dir',
                                help='Where to store training files.')
    argumentParser.add_argument('--profile', '-p',  default=None,
                                help='Optional configuration profile for music_extractor.')
    argumentParser.add_argument('--format', '-t', default='yaml', choices=['yaml', 'json'],
                                help='Store Yaml or Json files.')
    argumentParser.add_argument('--skip_analyzed', '-s', action='store_true',
                                help='Whether to skip already existing files.')
    argumentParser.add_argument('--store_frames', '-f', action='store_true',
                                help='Whether store analysis frames.')

    args = argumentParser.parse_args()

    extract_features(args.data_dir, args.exp_dir, profile=args.profile, skip_analyzed=args.skip_analyzed,
                     store_frames=args.store_frames, format=args.format)
