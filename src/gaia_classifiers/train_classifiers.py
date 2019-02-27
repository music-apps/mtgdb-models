#!/usr/bin/env python

import os
import yaml
import traceback

from argparse import ArgumentParser
from gaia2.scripts.classification import train_model
from gaia2.scripts.classification import json_to_sig

from path_config import DATASETS_GROUNDTRUTH, MTGDB_DIR


def get_files_in_dir(dirname, extension):
    for dirpath, dirnames, filenames in os.walk(dirname):
        for filename in [f for f in filenames if f.endswith(extension)]:
            yield os.path.join(dirpath, filename)


def match_groudtruth(dataset_data_dir, filelist_file, groundtruth_input_file,
                     groundtruth_output_file, tolerance=5):    
    # List .sig files 
    sig_files = [f for f in get_files_in_dir(
        dataset_data_dir, "sig")]

    ls = {}
    for f in sig_files:
        id = ('/'.join(f.split('/')[-1:])).replace('.sig', '')
        ls[id] = f

    # Get groundtruth entries
    if not os.path.exists(groundtruth_input_file):
        raise Exception('not valid metadata file for {}.'.format(
            groundtruth_input_file))

    gt_dict = yaml.load(open(groundtruth_input_file))
    gt = gt_dict['groundTruth']

    for key in gt.keys():
        gt[key.split('/')[-1]] = gt.pop(key)

    gt_keys = set(gt.keys())
    ls_keys = set(ls.keys())

    gt_tracks = len(gt_keys)

    gt_not_in_ls = len(gt_keys.difference(ls_keys))
    ls_not_in_gt = len(ls_keys.difference(gt_keys))

    gt_in_ls = gt_keys.intersection(ls_keys)

    gt_not_in_ls_per = 100 * gt_not_in_ls / gt_tracks

    if gt_not_in_ls == 0:
        gt_dict['groundTruth'] = gt
        print('Every track in the ground truth was found.')

    else:
        print('Warnning: {} ({}%) of the tracks in the ground truth are missing.'.format(
            gt_not_in_ls, gt_not_in_ls_per))

        gt_dict['groundTruth'] = dict(
            filter(lambda i: i[0] in gt_in_ls, gt.iteritems()))

    if ls_not_in_gt > 0:
        print('{} .sig files are not present in the groundtruth.'.format(ls_not_in_gt))
        ls = dict(filter(lambda i: i[0] in gt_in_ls, ls.iteritems()))

    # Write ground truth and filelist files
    with open(groundtruth_output_file, 'w') as f:
        yaml.dump(gt_dict, f)

    with open(filelist_file, 'w') as f:
        yaml.dump(ls, f)

    if gt_not_in_ls_per > tolerance:
        max_allowed = int(gt_tracks * tolerance // 100)
        raise Exception('{} ({}%) files from the groundtruth are not listed in the data directory. Maximun missing files allowed amount is {} ({}%).'.format(
            gt_not_in_ls, gt_not_in_ls_per, max_allowed, tolerance))


def train_models(data_dir, exp_dir, tolerance=5):
    for dataset, groundtruth_input_file in DATASETS_GROUNDTRUTH.items():
        try:
            dataset_data_dir = os.path.join(data_dir, dataset)
            dataset_exp_dir = os.path.join(exp_dir, dataset)

            filelist_file = os.path.join(dataset_exp_dir, "filelist.yaml")
            groundtruth_output_file = os.path.join(
                dataset_exp_dir, "groundtruth.yaml")
            project_file = os.path.join(
                dataset_exp_dir, "{}.project".format(dataset))
            results_model_file = os.path.join(
                dataset_exp_dir, "{}.history".format(dataset))
            log_file = os.path.join(dataset_exp_dir, "{}.log".format(dataset))

            if not os.path.exists(dataset_exp_dir):
                os.makedirs(dataset_exp_dir)

            open(log_file, 'w')

            print('\n\n{}\nProcessing "{}"\n{}\n\n"'.format(
                '*' * 30, dataset, '*' * 30))
            with open(log_file, 'a') as f:
                f.write('\n\n{}\nProcessing "{}"\n{}\n\n"'.format(
                    '*' * 30, dataset, '*' * 30))

            # Convert .json to .sig if required
            json_files = [f for f in get_files_in_dir(
                dataset_data_dir, "json")]
            yaml_files = [f for f in get_files_in_dir(dataset_data_dir, "sig")]

            if (len(json_files) > 0):
                yaml_files_existing = [f.rstrip('.sig') for f in yaml_files]

                yaml_files_missing = {os.path.splitext(os.path.basename(f))[0]: os.path.join(
                    dataset_data_dir, f) for f in json_files if f.rstrip('.json') not in yaml_files_existing}

                print("{} json files have to be converted into yamls. {} already exist.".format(
                    len(yaml_files_missing), len(yaml_files_existing)))

                filelist_to_convert_file = os.path.join(
                    dataset_exp_dir, "filelist-to-convert.yaml")

                yaml.dump(yaml_files_missing, open(
                    filelist_to_convert_file, "w"))
                json_to_sig.convertJsonToSig(
                    filelist_to_convert_file, filelist_file)

                if os.path.exists(filelist_to_convert_file):
                    os.remove(filelist_to_convert_file)

            groundtruth_input_file = os.path.join(
                MTGDB_DIR, 'stable', groundtruth_input_file)

            match_groudtruth(dataset_data_dir, filelist_file, groundtruth_input_file,
                             groundtruth_output_file, tolerance=tolerance)

            train_model.trainModel(groundtruth_output_file, filelist_file,
                                   project_file, dataset_exp_dir, results_model_file)

        except:
            print('Exception occured processing {}'.format(dataset))
            traceback.print_exc()

            with open(log_file, 'w') as f:
                f.write('Exception occured processing {}'.format(dataset))
                traceback.print_exc(file=f)


if __name__ == '__main__':
    argumentParser = ArgumentParser(
        "Trains the available Gaia classifiers given a path to the features.")
    argumentParser.add_argument('data_dir',
                                help='Directory with the .sig files.')
    argumentParser.add_argument('--exp_dir', '-o', default='exp/',
                                help='Directory to perform the experiments and store the results.')
    argumentParser.add_argument('--tolerance', '-t', default=5, type=float,
                                help='Allowed percentage of missing groundtruth files.')

    args = argumentParser.parse_args()

    train_models(args.data_dir, args.exp_dir, args.tolerance)
