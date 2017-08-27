import argparse
import gzip
import json
import os
import pickle
from collections import OrderedDict
import logging
import numpy as np


def read_vectors(path, unk_word=None, pad_word=None, max_vecs=1000000):
    vecs = OrderedDict()
    dim = 0
    with gzip.open(path, 'rb') if path.endswith('gz') else open(path, 'rb') as vector_file:
        for index, line in enumerate(vector_file):
            if not line.strip():
                continue
            if index >= max_vecs:
                break
            fields = line.split()
            vec = np.array([float(x) for x in fields[1:]], dtype=np.float32)
            vecs[fields[0]] = vec
            dim = vec.size
    if unk_word and unk_word not in vecs:
        vecs[unk_word] = np.random.normal(0, 0.01, dim)
    if pad_word and pad_word not in vecs:
        vecs[pad_word] = np.zeros(dim)
    return vecs, dim


def initialize_vectors(vector_map, vocabulary, dim):
    emb = np.zeros([len(vocabulary), dim], dtype=np.float32)
    for word, index in vocabulary.items():
        if word not in vector_map:
            vector_map[word] = np.random.normal(0, 0.01, dim)
        emb[index] = vector_map[word]
    return emb


def serialize(serializable, out_path, out_name=None):
    if out_name:
        out_name = out_name if out_name.endswith(".pkl") else "{}.pkl".format(out_name)
    path = os.path.join(out_path, out_name) if out_name else out_path
    with open(path, mode="wb") as out_file:
        pickle.dump(serializable, out_file)


def deserialize(in_path, in_name=None):
    if in_name:
        in_name = in_name if in_name.endswith(".pkl") else "{}.pkl".format(in_name)
    path = os.path.join(in_path, in_name) if in_name else in_path
    with open(path, mode="rb") as in_file:
        return pickle.load(in_file)


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def read_json(json_path):
    with open(json_path, 'r') as config:
        return json.load(config)


def configure_logger(log_path):
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-5.5s] %(message)s", datefmt='%m/%d/%Y %I:%M:%S %p')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)
