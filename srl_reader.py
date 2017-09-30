import os
from collections import defaultdict

START_OF_LABEL = "("
END_OF_LABEL = ")"
CONTINUATION = "*"


class ConllReader(object):
    def __init__(self, index_field_map):
        super(ConllReader, self).__init__()
        self._index_field_map = index_field_map

    def read_file(self, path):
        results = []
        with open(path) as conll_file:
            lines = []
            for line in conll_file:
                line = line.strip()
                if not line and lines:
                    results.append(self.read_instance([line.split() for line in lines]))
                    lines = []
                    continue
                lines.append(line)
            if lines:
                results.append(self.read_instance([line.split() for line in lines]))
        return results

    def read_instance(self, rows):
        return self.read_fields(rows)

    def read_files(self, path, extension):
        if os.path.isdir(path):
            results = []
            for input_file in sorted(os.listdir(path)):
                if input_file.endswith(extension):
                    results.extend(self.read_file(os.path.join(path, input_file)))
            return results
        else:
            return self.read_file(path)

    def read_fields(self, rows):
        sentence = defaultdict(list)
        for row in rows:
            for index, val in self._index_field_map.iteritems():
                sentence[val].append(row[index])
        return sentence


class ConllSrlReader(ConllReader):
    def __init__(self, index_field_map, pred_start, pred_end=0, phrase_index=None, pred_key="predicate"):
        super(ConllSrlReader, self).__init__(index_field_map)
        self._pred_start = pred_start
        self._pred_end = pred_end
        self._phrase_index = phrase_index
        self._pred_index = [key for key, val in self._index_field_map.items() if val == pred_key][0]

    def read_instance(self, rows):
        if self._phrase_index:
            return self.read_phrases(rows)
        else:
            return self.read_fields(rows), self.read_predicates(rows)

    def read_phrases(self, rows):
        phrases = []
        predicates = self.read_predicates(rows)
        chunk_preds = defaultdict(list)
        for i, (index, phrase) in enumerate(ConllSrlReader._read_chunks(rows, phrase_index=self._phrase_index)):
            phrases.append(self.read_fields(phrase))
            for pred_index, preds in predicates.items():
                chunk_preds[pred_index].append(preds[index])
        return phrases, chunk_preds

    def read_predicates(self, rows):
        pred_indices = []
        pred_cols = defaultdict(list)
        for i, row in enumerate(rows):
            predicate = row[self._pred_index]
            if predicate is not "-":
                pred_indices.append(i)
            for index in range(self._pred_start, len(row) - self._pred_end):
                pred_cols[index - self._pred_start].append(row[index])
        # convert from CoNLL05 labels to IOB labels
        for key, val in pred_cols.iteritems():
            pred_cols[key] = ConllSrlReader._convert_to_iob(val)
        # create predicate dictionary with keys as predicate word indices and values as corr. lists of labels (1 for each token)
        index = 0
        predicates = {}
        for i in pred_indices:
            predicates[i] = pred_cols[index]
            index += 1
        return predicates

    @staticmethod
    def _read_chunks(rows, phrase_index):
        chunked_rows = []
        curr_chunk = []
        curr_label = None
        index = None
        for i, row in enumerate(rows):
            prev_label = curr_label
            curr_label = row[phrase_index]
            if end_of_chunk(prev_label, curr_label):
                chunked_rows.append((index, curr_chunk))
                curr_chunk = []
            elif curr_chunk:
                curr_chunk.append(row)

            if start_of_chunk(prev_label, curr_label):
                index = i
                curr_chunk.append(row)
        if curr_chunk:
            chunked_rows.append((index, curr_chunk))
        return chunked_rows

    @staticmethod
    def _convert_to_iob(labels):
        def _get_label(_label):
            return _label.replace(START_OF_LABEL, "").replace(END_OF_LABEL, "").replace(CONTINUATION, "")

        current = None
        results = []
        for token in labels:
            if token.startswith(START_OF_LABEL):
                label = _get_label(token)
                results.append("B-" + label)
                current = label
            elif current and CONTINUATION in token:
                results.append("I-" + current)
            else:
                results.append("O")

            if token.endswith(END_OF_LABEL):
                current = None
        return results


class Conll2003Reader(ConllReader):
    def __init__(self, besio=False):
        super(Conll2003Reader, self).__init__({0: "word", 1: "pos", 2: "chunk", 3: "ne"})
        self.besio = besio

    def read_instance(self, rows):
        instance = super(Conll2003Reader, self).read_instance(rows)
        instance['ne'] = chunk(instance['ne'], besio=self.besio)
        return instance


class Conll2005Reader(ConllSrlReader):
    def __init__(self):
        super(Conll2005Reader, self).__init__(
            {0: "id", 1: "pos", 2: "parse", 3: "word", 4: "ne", 5: "roleset", 6: "predicate"}, 7)


class Conll2012Reader(ConllSrlReader):
    def __init__(self):
        super(Conll2012Reader, self).__init__(
            {3: "word", 4: "pos", 5: "parse", 6: "predicate", 7: "roleset"}, 11, 1)


class ConllPhraseReader(ConllSrlReader):
    def __init__(self):
        super(ConllPhraseReader, self).__init__(
            {0: "word", 1: "phrase", 2: "roleset", 3: "predicate"}, 4, phrase_index=1)


def end_of_chunk(prev, curr):
    prev_val, prev_tag = _get_val_and_tag(prev)
    curr_val, curr_tag = _get_val_and_tag(curr)
    if prev_val == 'O':
        return True
    if not prev_val:
        return False
    if prev_tag != curr_tag or prev_val == 'E' or curr_val == 'B' or curr_val == 'O' or prev_val == 'O':
        return True
    return False


def start_of_chunk(prev, curr):
    prev_val, prev_tag = _get_val_and_tag(prev)
    curr_val, curr_tag = _get_val_and_tag(curr)
    if prev_tag != curr_tag or curr_val == 'B' or curr_val == 'O':
        return True
    return False


def _get_val_and_tag(label):
    if not label:
        return '', ''
    if label == 'O':
        return label, ''
    return label.split('-')


def chunk(labeling, besio=True, conll=False):
    if conll:
        besio = True
    result = []
    prev_type = None
    curr = []
    for label in labeling:
        if label == 'O' or label == '<UNK>':
            state, chunk_type = 'O', ''
        else:
            split_index = label.index('-')
            state, chunk_type = label[:split_index], label[split_index + 1:]
        if state == 'I' and chunk_type != prev_type:  # new chunk of different type
            state = 'B'
        if state in 'OB' and curr:  # end of chunk
            result += to_besio(curr) if besio else curr
            curr = []
        if state == 'O':
            result.append(state)
        else:
            curr.append(state + "-" + chunk_type)
        prev_type = chunk_type
    if curr:
        result += to_besio(curr) if besio else curr
    if conll:
        result = [to_conll(label) for label in result]
    return result


def to_besio(iob_labeling):
    if len(iob_labeling) == 1:
        return ['S' + iob_labeling[0][1:]]
    return iob_labeling[:-1] + ['E' + iob_labeling[-1][1:]]


def to_conll(iob_label):
    label_type = get_type(iob_label)
    if iob_label.startswith("B-"):
        return "(" + label_type + "*"
    if iob_label.startswith("S-"):
        return "(" + label_type + "*)"
    if iob_label.startswith("E-"):
        return "*)"
    return "*"


def get_type(iob_label):
    return iob_label.replace("B-", "").replace("E-", "").replace("S-", "").replace("I-", "")
