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
### Prepare Training Data
```bash
python srl_feature_extractor.py --mode new --input data/datasets/conll2005/train --output data/datasets/conll2005/train.pkl --vocab data/datasets/conll2005 --config data/configs/he_acl_2017.json
python srl_feature_extractor.py --mode update --input data/datasets/conll2005/dev --output data/datasets/conll2005/dev.pkl --vocab data/datasets/conll2005 --config data/configs/he_acl_2017.json
python srl_feature_extractor.py --mode update --input data/datasets/conll2005/test --output data/datasets/conll2005/test.pkl --vocab data/datasets/conll2005 --config data/configs/he_acl_2017.json

```
### Train Model
```bash
python srl_trainer.py --train data/datasets/conll2005/train.pkl --valid data/datasets/conll2005/dev.pkl --vocab data/datasets/conll2005 --script data/scripts/srl-eval.pl --config data/configs/he_acl_2017.json --save data/models/conll2005 --log data/logs/log.txt
```