import argparse

import torch as th
import numpy as np
import torch
import torch.nn as nn



      #将字符串转换为布尔值。
def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")



def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)

def args_to_dict(args, keys):
    return {k: getattr(args, k) for k in keys}

def select_batch_size(args, dataset):


    if dataset == 'Pemsd7L_288':
        batch_size = int(args.batch_size_pemsd7L * args.batch_ratio)
    elif dataset == 'Pemsd7_288':
        batch_size = int(args.batch_size_pemsd7 * args.batch_ratio)
    elif dataset == 'PEMS08_288':
        batch_size = int(args.batch_size_PEMS08 * args.batch_ratio)
    elif dataset == 'PemsBay_288':
        batch_size = int(args.batch_size_PemsBay * args.batch_ratio)
    elif 'TaxiNYC' in dataset:
        batch_size = int(args.batch_size_nyc * args.batch_ratio)
    elif 'TaxiBJ' in dataset:
        batch_size = int(args.batch_size_taxibj * args.batch_ratio)
    elif 'Graph' in dataset:
         batch_size = int(args.batch_size_graph_large * args.batch_ratio)

    # if 'Graph' in dataset:
    #     batch_size = int(args.batch_size_graph_large * args.batch_ratio)
    #     if 'SH' in dataset:
    #          batch_size = int(batch_size*0.8)
    #     elif 'Pemsd7' in dataset:
    #          batch_size = int(args.batch_size_pemsd7 * args.batch_ratio)
    return batch_size

def select_patch_size(args, data):


    patch_size = args.patch_size
    if 'Graph' in data:
        if 'SH' in data:
            patch_size = 100
        elif 'Pems' in data:
            patch_size=args.patch_size  ##Pemsd7L:3

    # MIN, MID, MAX = [2,2,100]
    # patch_size = MIN
    # if 'TaxiBJ' in data or 'Flow' in data or 'TaxiNYC' in data or 'Crowd' in data or 'Pop' in data:
    #     patch_size = MIN
    #
    # if 'Graph' in data:
    #     patch_size = MAX
    #     if 'SH' in data:
    #         patch_size = MAX * 2
    #     elif 'Pems' in data:
    #         patch_size=3  ##Pemsd7L:3

    return patch_size


def tensor_to_adjacency_matrix(tensor):
    num_rows, num_cols = tensor.shape
    adjacency_matrix = torch.zeros((num_rows * num_cols, num_rows * num_cols), dtype=torch.int)

    # Define neighbors offset
    neighbors_offset = [(0, 1), (1, 0), (0, -1), (-1, 0),(0,0)]

    # Iterate through each element in the tensor
    for i in range(num_rows):
        for j in range(num_cols):
            current_index = i * num_cols + j

            # Check neighbors
            for offset_i, offset_j in neighbors_offset:
                new_i, new_j = i + offset_i, j + offset_j
                if 0 <= new_i < num_rows and 0 <= new_j < num_cols:
                    neighbor_index = new_i * num_cols + new_j
                    adjacency_matrix[current_index][neighbor_index] = 1

    return adjacency_matrix