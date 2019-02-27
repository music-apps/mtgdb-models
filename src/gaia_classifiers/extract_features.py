#!/usr/bin/env python2.7

import os

from argparse import ArgumentParser
from essentia.pytools.extractors import batch_music_extractor
from path_config import DATASETS_DATA, MTGDB_DIR


def extract_features(data_dir, profile=None,
                     skip_analyzed=True, store_frames=False, format='yaml'):
    for dataset, data_paths in DATASETS_DATA.iteritems():
        print('Processing "{}"'.format(dataset))

        dataset_out_dir = os.path.join(data_dir, dataset)

        for data_path in data_paths:
            dataset_in_dir = os.path.join(MTGDB_DIR, 'stable', data_path)

            batch_music_extractor(dataset_in_dir, dataset_out_dir,
                                  profile=profile, skip_analyzed=skip_analyzed,
                                  store_frames=store_frames, format=format)


if __name__ == '__main__':
    argumentParser = ArgumentParser(
        'Generate music_extractor features for all the MTGDB datasets relevant to train Gaia models.')
    argumentParser.add_argument('data_dir',
                                help='Where to store the .sig files.')
    argumentParser.add_argument('--profile', '-p',  default=None,
                                help='Optional configuration profile for music_extractor.')
    argumentParser.add_argument('--format', '-t', default='yaml', choices=['yaml', 'json'],
                                help='Store Yaml or Json files.')
    argumentParser.add_argument('--skip_analyzed', '-s', action='store_true',
                                help='Whether to skip already existing files.')
    argumentParser.add_argument('--store_frames', '-f', action='store_true',
                                help='Whether store analysis frames.')

    args = argumentParser.parse_args()

    extract_features(args.data_dir, profile=args.profile, skip_analyzed=args.skip_analyzed,
                     store_frames=args.store_frames, format=args.format)
