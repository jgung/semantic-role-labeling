#  Deep Semantic Role Labeling in Tensorflow

This repository contains the following:

* A Tensorflow implementation of a deep SRL model based on the architecture described in:
[Deep Semantic Role Labeling: What works and what's next](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf)
* Deep semantic role labeling experiments using phrase-constrained models and subword (character-level) features

## Getting Started
### Prerequisites
* Python 3
* virtualenv
### Virtualenv Installation
```bash
virtualenv ~/.venvs/tf-srl
source ~/.venvs/tf-srl/bin/activate
cd semantic-role-labeling
pip install -r requirements.txt
```
### Download word embeddings
We use [GloVe](https://nlp.stanford.edu/projects/glove/) 100-dimensional vectors trained on 6B tokens. They can be downloaded with the following:
```bash
./data/scripts/get-resources.sh
```
## Usage
### Training CoNLL-2005
In order to generate SRL training data for [CoNLL-2005](http://www.lsi.upc.edu/~srlconll/soft.html), you will need to download
and extract the PTB corpus [LDC99T42](https://catalog.ldc.upenn.edu/ldc99t42) (which is not publicly available).

To train a model based on [Deep Semantic Role Labeling: What works and what's next](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf)
(He et al. 2017), you can then use the following scripts:
```bash
# download and prepare training data (only needs to be run once)
./data/scripts/conll05-data.sh -i /path/to/ptb/
# extract features and train default model with CoNLL-2005 train/devel split
./data/scripts/train-srl.sh -i data/datasets/conll05/ -o data/experiments/conll05/
```
To train a phrase-constrained model, you need to override the default configuration file and mode:
```bash
./data/scripts/train-srl.sh -i data/datasets/conll05/ -o data/experiments/conll05-phrase/ -c data/configs/phrase.json -m phrase
```
### Training CoNLL-2012
In order to generate SRL training data corresponding to the train-dev-test split from [CoNLL-2012](http://cemantix.org/data/ontonotes.html), you will need to download
and extract OntoNotes 5 [LDC2013T19](https://catalog.ldc.upenn.edu/ldc2013t19).

Having done this, you can train a model as follows:
```bash
# download and prepare data (only needs to be run once)
./data/scripts/conll2012-data.sh -i /path/to/ontonotes-release-5.0/
# extract features and train default model with CoNLL-2012 train/devel split
./data/scripts/train-srl.sh -i data/datasets/conll2012/ -o data/experiments/conll2012/
```
### Training with other data
It's possible to train using CoNLL-style data in other formats (with different columns). To do this, you must specify a few
required fields through a .json configuration file:
```json
{
  "columns": {
    "word": 0, 
    "roleset": 4,
    "predicate": 5
  },
  "arg_start_col": 6
}
```
Here, `"word": 0` means that words appear in the first column. Similarly, `"roleset": 4` means that the roleset or sense
for predicates appears in the 4th column. `"predicate"` provides the column index of the lemma of the predicate.
Other columns can be added for use in feature extraction, but these are the bare minimum required.
`"arg_start_col"` gives the first column containing argument labels. No additional columns can occur after argument columns.

Then, if you have a training file named `train.conll` and dev file named `valid.conll` in `path/to/data/directory`,
you can train as follows with a custom reader named `reader.json`:
```bash
./data/scripts/train-srl.sh -i path/to/data/directory -o path/to/output/directory -t train.conll -v valid.conll --custom reader.json
```

### Evaluation
To simplify evaluation, `train-srl.sh` can be used directly. For CoNLL-05, for example, you can test on the Brown corpus as follows:
```bash
./data/scripts/train-srl.sh -i data/datasets/conll05/ -o data/experiments/conll05/ --test test-brown.conll
```
where `test-brown.conll` must be located in `data/datasets/conll05/`.