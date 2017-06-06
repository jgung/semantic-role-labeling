from collections import defaultdict

START_OF_LABEL = "("
END_OF_LABEL = ")"
CONTINUATION = "*"


class ConllReader(object):
    def __init__(self, index_field_map, pred_start, pred_end=0):
        super(ConllReader, self).__init__()
        self._index_field_map = index_field_map
        self._pred_start = pred_start
        self._pred_end = pred_end

    def read_file(self, path):
        results = []
        with open(path) as conll_file:
            lines = []
            for line in conll_file:
                line = line.strip()
                if not line and lines:
                    results.append(self.read_sentence(lines))
                    lines = []
                    continue
                lines.append(line)
        return results

    def read_sentence(self, lines):
        sentence = defaultdict(list)
        pred_cols = defaultdict(list)
        rows = [line.split() for line in lines]
        for row in rows:
            for index, val in self._index_field_map.iteritems():
                sentence[val].append(row[index])
            for index in range(self._pred_start, len(row) - self._pred_end):
                pred_cols[index - self._pred_start].append(row[index])
        for key, val in pred_cols.iteritems():
            pred_cols[key] = ConllReader._convert_to_iob(val)

        index = 0
        predicates = {}
        for i, pred in enumerate(sentence["predicate"]):
            if pred is not "-":
                predicates[i] = pred_cols[index]
                index += 1

        return sentence, predicates

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


class Conll2005Reader(ConllReader):
    def __init__(self):
        super(Conll2005Reader, self).__init__(
            {0: "id", 1: "pos", 2: "parse", 3: "word", 4: "ne", 5: "roleset", 6: "predicate"}, 7)


def get_type(iob_label):
    return iob_label.replace("B-", "").replace("E-", "").replace("S-", "").replace("I-", "")


def chunk(labeling, besio=True, conll=False):
    if conll:
        besio = True
    result = []
    prev_type = None
    curr = []
    for label in labeling:
        if label == 'O':
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
