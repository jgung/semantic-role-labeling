import numpy as np

from srl.common.constants import UNKNOWN_WORD


def combine_embeddings(groupings, vector_dict):
    """
    Compute mean vectors for given groups of words given a word embedding dictionary.
    :param groupings: groups of words as a dictionary from group names onto word lists
    :param vector_dict: word embedding dictionary
    :return: dictionary of mean embeddings
    """
    results = {}
    for name, words in groupings.items():
        vectors = []
        for word in words:
            if word in vector_dict:
                vectors.append(vector_dict[word])
        if not vectors:
            vectors.append(vector_dict[UNKNOWN_WORD])
        result = np.average(vectors, axis=0)
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        results[name] = result
    return results


def output_vectors(vector_dict, output_path):
    """
    Writes word vectors to a provided path, in the standard format:
    `word1 dim1 dim2 dim3 dim4...dimN
     word2 dim1 dim2 dim3 dim4...dimN
     ...`
    :param vector_dict: dictionary from words to 1d Numpy arrays
    :param output_path: path to file to save resulting vectors
    """
    with open(output_path, 'w') as output_file:
        for word, vector in vector_dict.items():
            output_file.write(word + ' ' + ' '.join([str(value) for value in vector]) + '\n')
