#  Deep Semantic Role Labeling in Tensorflow

This repository contains the following:

* A Tensorflow implementation of a deep SRL model based on the architecture described in:
[Deep Semantic Role Labeling: What works and what's next](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf)
* Deep semantic role labeling experiments using phrase-constrained models and subword (character-level) features

## Getting Started
### Prerequisites
* Python 2.7
* virtualenv
### Virtualenv Installation
```bash
virtualenv ~/.virtualenvs/tf
source ~/.virtualenvs/tf/bin/activate
cd semantic-role-labeling
pip install -r requirements.txt
```
## Usage
### Training CoNLL-2005
In order to generate SRL training data for [CoNLL-2005](http://www.lsi.upc.edu/~srlconll/soft.html), you will need to download
and extract the PTB corpus [LDC99T42](https://catalog.ldc.upenn.edu/ldc99t42) (which is not publicly available).

To train a model based on [Deep Semantic Role Labeling: What works and what's next](https://homes.cs.washington.edu/~luheng/files/acl2017_hllz.pdf)
(He et al. 2017), you can then use the following scripts:
```bash
# download and prepare training data (only needs to be run once)
./data/scripts/conll05-data.sh /path/to/ptb/
# extract features and train default model with CoNLL05 train/devel split
./data/scripts/conll05-train.sh data/datasets/conll05/ data/experiments/conll05/
```