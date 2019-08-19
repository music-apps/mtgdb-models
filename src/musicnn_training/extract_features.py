import os
import yaml
from subprocess import call

from argparse import ArgumentParser
from path_config import DATASETS_DATA, DATASETS_GROUNDTRUTH, MTGDB_DIR, MUSICNN_DIR


def extract_features(data_dir, exp_dir, profile=None,
                     skip_analyzed=True, store_frames=False, format='yaml'):
    for dataset, data_paths in DATASETS_DATA.items():
        print('Processing "{}"'.format(dataset))

        dataset_out_dir = os.path.join(data_dir, dataset)
        exp_out_dir = os.path.join(exp_dir, dataset)

        index_file = os.path.join(dataset_out_dir, 'index.tsv')
        gt_train = os.path.join(exp_dir, dataset, 'gt_train.csv')
        gt_val = os.path.join(exp_dir, dataset, 'gt_val.csv')

        os.makedirs(dataset_out_dir, exist_ok=True)
        os.makedirs(exp_out_dir, exist_ok=True)

        for audio_path in data_paths:
            dataset_in_dir = os.path.join(MTGDB_DIR, 'stable', audio_path)

            groundtruth_input_file = os.path.join(
                MTGDB_DIR, 'stable', DATASETS_GROUNDTRUTH[dataset])

            groundtruth = yaml.load(open(groundtruth_input_file))

            audio_paths = ['{:03d}\t{}'.format(idx, path) for idx, path in enumerate(groundtruth['groundTruth'].keys())]

            with open(index_file, 'w') as f:
                f.write('\n'.join(audio_paths))

            config_file_template = open(os.path.join(MUSICNN_DIR, 'src', 'config_file_template.yaml')).read()

            config_file = os.path.join(MUSICNN_DIR, 'src', 'config_file.yaml')
            config_file_log = os.path.join(exp_out_dir, 'config_file.yaml')

            configured_file = config_file_template % {'dataset': dataset,
                                                      'data_folder': os.path.abspath(dataset_out_dir) + '/',
                                                      'audio_folder': os.path.abspath(dataset_in_dir) + '/',
                                                      'identifier': dataset,
                                                      'index_file': 'index.tsv',
                                                      'gt_train': os.path.abspath(gt_train),
                                                      'gt_val': os.path.abspath(gt_val)
                                                      }

            # write the project file
            with open(config_file_log, 'w') as pfile:
                pfile.write(configured_file)

            # write the project file
            with open(config_file, 'w') as pfile:
                pfile.write(configured_file)

            script = os.path.join(MUSICNN_DIR, 'src', 'preprocess_librosa.py')
            call(['python', script, 'mtgdb_spec'], cwd=os.path.dirname(script))


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
