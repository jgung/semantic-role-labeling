import argparse
import gzip
import numpy as np
import pickle
import os
from collections import OrderedDict
from srl_reader import Conll2005Reader


PAD_WORD = "<PAD>"
UNKNOWN_WORD = "<UNK>"
PAD_INDEX = 0
UNKNOWN_INDEX = 1


def read_vectors(path, max_vecs=1000000, normalize=True):
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
            if normalize:
                vec = vec / np.linalg.norm(vec)
            vecs[fields[0]] = vec
            dim = vec.size
    if UNKNOWN_WORD not in vecs:
        vecs[UNKNOWN_WORD] = np.zeros(dim)
    if PAD_WORD not in vecs:
        vecs[PAD_WORD] = np.zeros(dim)
    return vecs, dim


def initialize_vectors(vector_map, vocabulary, dim, normalize=True):
    emb = np.zeros([len(vocabulary), dim], dtype=np.float32)
    for word, index in vocabulary.items():
        if word in vector_map:
            emb[index] = vector_map[word]
        else:
            emb[index] = np.random.uniform(-1, 1, dim)
            emb[index] = emb[index] / np.linalg.norm(emb[index]) if normalize else emb[index]
    return emb


def create_dictionaries(reader, path, label_dict=None, word_dict=None):
    if word_dict is None:
        word_dict = {PAD_WORD: PAD_INDEX, UNKNOWN_WORD: UNKNOWN_INDEX}
    if label_dict is None:
        label_dict = {PAD_WORD: PAD_INDEX}
    sentences = reader.read_file(path)
    for sentence, predicates in sentences:
        for word in sentence['word']:
            if word not in word_dict:
                word_dict[word] = len(word_dict)
        for key, predicate in predicates.iteritems():
            for label in predicate:
                if label not in label_dict:
                    label_dict[label] = len(label_dict)
    return label_dict, word_dict


def create_instances(reader, path, label_dict, word_dict):
    sentences = reader.read_file(path)
    instances = []
    for sentence, predicates in sentences:
        words = [word_dict.get(word, word_dict.get(UNKNOWN_WORD)) for word in sentence['word']]
        for key, predicate in predicates.iteritems():
            labels = [label_dict[pred] for pred in predicate]
            instances.append({"index": key,
                              "is_predicate": [index == key and 1 or 0 for index in range(0, len(words))],
                              "words": np.asarray(words, dtype=np.int32),
                              "labels": np.asarray(labels, dtype=np.int32),
                              "length": len(words)})
    return instances


def write_model_files(out_path, word_vectors, words, labels):
    with open(os.path.join(out_path, "vectors.pkl"), mode="wb") as vec_file:
        pickle.dump(word_vectors, vec_file)
    with open(os.path.join(out_path, "words.pkl"), mode="wb") as words_file:
        pickle.dump(words, words_file)
    with open(os.path.join(out_path, "labels.pkl"), mode="wb") as labels_file:
        pickle.dump(labels, labels_file)


def load_model_files(input_path):
    with open(os.path.join(input_path, "vectors.pkl"), mode="rb") as vec_file:
        vectors = pickle.load(vec_file)
    with open(os.path.join(input_path, "words.pkl"), mode="rb") as words_file:
        words = pickle.load(words_file)
    with open(os.path.join(input_path, "labels.pkl"), mode="rb") as labels_file:
        labels = pickle.load(labels_file)
    return vectors, words, labels


def write_instances(output_path, data):
    with open(os.path.join(output_path), mode="wb") as data_file:
        pickle.dump(data, data_file)


def load_instances(input_path):
    with open(os.path.join(input_path), mode="rb") as data_file:
        result = pickle.load(data_file)
    return result


def main(flags):
    if flags.load:
        vectors, vec_dim = read_vectors(flags.vectors)
        labels_dict, words_dict = create_dictionaries(Conll2005Reader(), flags.input)
        embedding = initialize_vectors(vector_map=vectors, vocabulary=words_dict, dim=vec_dim)
        write_model_files(flags.vocab, embedding, words_dict, labels_dict)
    else:
        _, words_dict, labels_dict = load_model_files(flags.vocab)
    instances = create_instances(Conll2005Reader(), flags.input, labels_dict, words_dict)
    write_instances(flags.input + ".pkl", instances)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default=True, type=bool,
                        help='Create new vocabularies and vector file using input data.')
    parser.add_argument('--input', required=True, type=str, help='CoNLL-formatted input file path.')
    parser.add_argument('--vocab', required=True, type=str, help='Vocab directory path.')
    parser.add_argument('--vectors', required=False, type=str, help='Path to word vectors.')
    main(parser.parse_args())
