import os
import yaml
from subprocess import call

from argparse import ArgumentParser
from path_config import DATASETS_DATA, DATASETS_GROUNDTRUTH, MTGDB_DIR, MUSICNN_DIR

import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold


def audio_to_repr_path(audio_path, dataset):
    audio_path = '.'.join(audio_path.split('.')[:-1]) + '.pk'
    return 'audio_representation/{}__time-freq/{}'.format(dataset, audio_path)


def rosamerica_map(label):
    if label == 'classical':
        return 'cla'
    elif label == 'electronic':
        return 'dan'
    elif label == 'hip hop':
        return 'hip'
    elif label == 'jazz':
        return 'jaz'
    elif label == 'pop':
        return 'pop'
    elif label == 'rhythm and blues':
        return 'rhy'
    elif label == 'rock':
        return 'roc'
    else:
        return label


def tzanetakis_map(label):
    if label == 'classical':
        return 'cla'
    elif label == 'blues':
        return 'blu'
    elif label == 'country':
        return 'cou'
    elif label == 'hip hop':
        return 'hip'
    elif label == 'jazz':
        return 'jaz'
    elif label == 'pop':
        return 'pop'
    elif label == 'ska':
        return 'reg'
    elif label == 'rock':
        return 'roc'
    else:
        return label


def dortmund_map(label):
    if label == 'blues':
        return 'blues'
    elif label == 'electronic':
        return 'electronic'
    elif label == 'folk':
        return 'folkcountry'
    elif label == 'country':
        return 'folkcountry'
    elif label == 'rhythm and blues':
        return 'funksoulrnb'
    elif label == 'jazz':
        return 'jazz'
    elif label == 'pop':
        return 'pop'
    elif label == 'hip hop':
        return 'raphiphop'
    elif label == 'rock':
        return 'rock'
    else:
        return label


def ccetl(args):
    data_dir = args.data_dir
    groundtruth_file = args.groundtruth_file
    index_file = args.index_file
    index_basedir = args.index_basedir
    skip_analyzed = args.skip_analyzed
    with_gpu = args.with_gpu
    features = args.features
    evaluation = args.evaluation
    exp_dir = args.exp_dir
    lib = args.lib
    transfer_learning = args.transfer_learning

    config_file_template = open(os.path.join(MUSICNN_DIR, 'src', 'config_file_template.yaml')).read()

    if features:
        script = os.path.join(MUSICNN_DIR, 'src', 'preprocess_crosseval_essentia.py')
        call(['python', script, index_file, index_basedir, data_dir, lib],
             cwd=os.path.dirname(script))

    for dataset, data_paths in DATASETS_DATA.items():
        print('Processing "{}"'.format(dataset))

        dataset_out_dir = os.path.join(data_dir, dataset)

        groundtruth_out_file = os.path.join(dataset_out_dir, 'groundtruth.tsv')

        os.makedirs(dataset_out_dir, exist_ok=True)

        for audio_path in data_paths:
            dataset_in_dir = os.path.join(MTGDB_DIR, 'stable', audio_path)

            groundtruth_input_file = os.path.join(
                MTGDB_DIR, 'stable', DATASETS_GROUNDTRUTH[dataset])

            groundtruth = yaml.load(open(groundtruth_input_file), Loader=yaml.SafeLoader)

            paths = groundtruth['groundTruth'].keys()
            labels = groundtruth['groundTruth'].values()
            ids = list(range(len(labels)))

            genres = np.array(list(set(groundtruth['groundTruth'].values())))
            genres = genres.reshape(-1, 1)

            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(genres)

            # load groundtruth_crosseval
            with open(groundtruth_file) as f:
                ids = []
                labels = []
                for line in f.readlines():
                    line = line.rstrip().split('\t')

                    ids.append(line[0])
                    labels.append(eval(line[1])[0])

            if dataset == 'genre_rosamerica':
                labels = list(map(rosamerica_map, labels))

            elif dataset == 'genre_dortmund':
                labels = list(map(dortmund_map, labels))

            elif dataset == 'genre_tzanetakis':
                labels = list(map(tzanetakis_map, labels))
            else:
                raise('dataset not available')

            labels = np.array(labels)
            labels = list(enc.transform(labels.reshape(-1, 1)).toarray())

            groundtruth_out = ['{}\t{}'.format(id, list(label)) for id, label in zip(ids, labels)]
            with open(groundtruth_out_file, 'w') as f:
                f.write('\n'.join(groundtruth_out))

            if evaluation:
                # call training script
                with open(os.path.join(exp_dir, dataset, 'experiment_id_whole')) as f:
                    model = f.read().rstrip()

                model_fol = os.path.join(exp_dir, dataset, 'experiments')

                results_file = os.path.join(dataset_out_dir, 'results')
                predictions_file = os.path.join(dataset_out_dir, 'predictions.json')

                script = os.path.join(MUSICNN_DIR, 'src', 'evaluate_crosseval.py')
                if transfer_learning:
                    call(['python', script, groundtruth_out_file, index_file,
                          model_fol, results_file, predictions_file, '--transfer_learning', '-l', model],
                         cwd=os.path.dirname(script))

                else:
                    call(['python', script, groundtruth_out_file, index_file,
                          model_fol, results_file, predictions_file, '-l', model],
                         cwd=os.path.dirname(script))


if __name__ == '__main__':
    parser = ArgumentParser(
        'Generate music_extractor features for all the MTGDB datasets relevant to train Gaia models.')
    parser.add_argument('data_dir',
                        help='Where to store the .npy files.')
    parser.add_argument('exp_dir',
                        help='Where to store the .npy files.')
    parser.add_argument('groundtruth_file',
                        help='Where to store the .npy files.')
    parser.add_argument('index_file',
                        help='Where to store the .npy files.')
    parser.add_argument('index_basedir',
                        help='Where to store the .npy files.')
    parser.add_argument('lib', choices=['essentia', 'librosa'],
                        help='dsp lib')
    parser.add_argument('--transfer_learning', action='store_true')
    parser.add_argument('--skip_analyzed', '-s', action='store_true',
                        help='Whether to skip already existing files.')
    parser.add_argument('--with_gpu', action='store_true',
                        help='Wether GPU resources are available')
    parser.add_argument('--features', '-f', action='store_true',
                        help='Wether to do feature extraction.')
    parser.add_argument('--evaluation', '-e', action='store_true',
                        help='Wether to do evaluation.')

    args = parser.parse_args()

    ccetl(args)
