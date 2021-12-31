import argparse
import logging
from os import mkdir, path
from pathlib import Path
import h5py
import numpy as np
import torch
import collections.abc as collections
import shutil
from .utils.parsers import parse_image_lists
from .utils.read_write_model import read_images_binary
from .utils.io import list_h5_names


def main(descriptors, output, num_matched,
         images, query, db_descriptors=None):
    logging.info('Extracting image pairs from a retrieval database.')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    db_names = list_h5_names(db_descriptors)
    query_names = list_h5_names(descriptors)

    def get_descriptors(names, path, name2idx=None, key='global_descriptor'):
        if name2idx is None:
            with h5py.File(str(path), 'r') as fd:
                desc = [fd[n][key].__array__() for n in names]
        else:
            desc = []
            for n in names:
                with h5py.File(str(path[name2idx[n]]), 'r') as fd:
                    desc.append(fd[n][key].__array__())
        return torch.from_numpy(np.stack(desc, 0)).to(device).float()

    db_desc = get_descriptors(db_names, db_descriptors)
    query_desc = get_descriptors(query_names, descriptors)
    sim = torch.einsum('id,jd->ij', query_desc, db_desc)
    topk = torch.topk(sim, num_matched, dim=1).indices.cpu().numpy()

    pairs = []

    shutil.rmtree(str(output.parent)+'/matches',ignore_errors=True)
    mkdir(str(output.parent)+'/matches')

    for query, indices in zip(query_names, topk):
        query_match_folder = str(output.parent)+'/matches/'+query.split('.')[0]
        mkdir(query_match_folder)
        for i in indices:
            pair = (query, db_names[i])
            pairs.append(pair)
            if images is not None:
                shutil.copy(path.join(str(images), db_names[i]), path.join(query_match_folder, db_names[i]))

    logging.info(f'Found {len(pairs)} pairs.')
    with open(output, 'w') as f:
        f.write('\n'.join(' '.join([i, j]) for i, j in pairs))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--descriptors', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--num_matched', type=int, required=True)
    parser.add_argument('--query_prefix', type=str, nargs='+')
    parser.add_argument('--query_list', type=Path)
    parser.add_argument('--db_prefix', type=str, nargs='+')
    parser.add_argument('--db_list', type=Path)
    parser.add_argument('--db_model', type=Path)
    parser.add_argument('--db_descriptors', type=Path)
    args = parser.parse_args()
    main(**args.__dict__)
