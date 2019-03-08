#!/usr/bin/env python
"""
Usage:
    python split_data.py --path ./data/PSPOption_RAW_Data.pkl --train_size 0.7 --valid_size 0.15 --test_size 0.15 --out ./data/PSPOption_RAW_Data

Runnig this will create 3 files:
    ./data/PSPOption_RAW_Data_TRAIN.pkl
    ./data/PSPOption_RAW_Data_VALID.pkl
    ./data/PSPOption_RAW_Data_TEST.pkl

"""
import argparse
import random
import pandas
import utils.file_utils as file_utils


def argparser():
    """
    Command line argument parser
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str)
    parser.add_argument('--train_size', type=float, default=0.)
    parser.add_argument('--valid_size', type=float, default=0.)
    parser.add_argument('--test_size', type=float, default=0.)
    parser.add_argument('--out', type=str)
    return parser.parse_args()


def get_split(args):
    assert args.train_size + args.valid_size + args.test_size == 1.
    split = {}
    if args.train_size > 0.:
        split['train'] = args.train_size
    if args.valid_size > 0.:
        split['valid'] = args.valid_size
    if args.test_size > 0.:
        split['test'] = args.test_size
    return split


def _get_subset_sizes(data_size, split):
    n_subset = len(split)
    subset_sizes = {}
    subset_counter = 0
    data_counter = 0
    for subset_name, subset_pct in split.items():
        subset_counter += 1
        if subset_counter == n_subset:
            subset_sizes[subset_name] = data_size - data_counter
        else:
            subset_sizes[subset_name] = round(data_size * subset_pct)
        data_counter += subset_sizes[subset_name]
    assert sum(subset_sizes.values()) == data_size
    return subset_sizes


def _get_split_idx(data_size, split):
    """
    Shuffle the index of the data to split
    into train/valid/test indices
    """
    examples = range(data_size)
    subset_sizes = _get_subset_sizes(data_size, split)
    split_idx = {}
    for subset_name, subset_size in subset_sizes.items():
        split_idx[subset_name] = random.sample(examples, subset_size)
        examples = [example for example in examples if example not in split_idx[subset_name]]
    return split_idx


def split_data(dataframe, split):
    data_size = len(dataframe)
    split_idx = _get_split_idx(data_size, split)
    data = {k: dataframe.iloc[v] for k, v in split_idx.items()}
    return data


def save_data(dataframe, name, out_dir):
    path = '{}{}.csv'.format(out_dir, name)
    dataframe.to_csv(path, index=False, header=False)


def main(args):
    split = get_split(args)
    data = file_utils.pickle2dataframe(args.path)
    data = split_data(data, split)
    for data_name, data in data.items():
        out_path = '{}_{}.pkl'.format(args.out, data_name.upper())
        file_utils.dataframe2pickle(data, out_path)


if __name__ == '__main__':
    main(argparser())
