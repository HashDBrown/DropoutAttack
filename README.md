# Dropout Attacks

This is the official repo for ["Dropout Attacks"](https://www.computer.org/csdl/proceedings-article/sp/2024/313000a026/1RjEa2qP0fm) by Andrew Yuan, Alina Oprea, and Cheng Tan

## Table of Contents
- [Installing](#installing)
- [Replication](#replication)
- [Citation](#citation)


## Installing
```
git clone git@github.com:awyuan/dropout-attacks.git # [TODO: FILL IN REPO HERE]
cd dropout-attacks
pip install -r requirements.txt
```

## Replication
To replicate the results reported in the paper, run `replication.sh` from the dropout-attacks folder. If only running
parts of the script, make sure you run any commands from within the evaluation folder. To generate figure pdfs found in
`./paper/figures_charts/`, run `create_figures.sh` from the dropout-attacks folder. 

## Citation
If you have found this project to be useful, please consider citing:

```
@INPROCEEDINGS {,
author = {A. Yuan and A. Oprea and C. Tan},
booktitle = {2024 IEEE Symposium on Security and Privacy (SP)},
title = {Dropout Attacks},
year = {2024},
volume = {},
issn = {2375-1207},
pages = {26-26},
abstract = {Dropout is a common operator in deep learning, aiming to prevent overfitting by randomly dropping neurons during training. This paper introduces a new family of poisoning attacks against neural networks named DROPOUTATTACK. DROPOUTATTACK attacks the dropout operator by manipulating the selection of neurons to drop instead of selecting them uniformly at random. We design, implement, and evaluate four DROPOUTATTACK variants that cover a broad range of scenarios. These attacks can slow or stop training, destroy prediction accuracy of target classes, and sabotage either precision or recall of a target class. In our experiments of training a VGG-16 model on CIFAR-100, our attack can reduce the precision of the victim class by 34.6% (81.7% → 47.1%) without incurring any degradation in model accuracy},
keywords = {ml security;ml attacks},
doi = {10.1109/SP54263.2024.00026},
url = {https://doi.ieeecomputersociety.org/10.1109/SP54263.2024.00026},
publisher = {IEEE Computer Society},
address = {Los Alamitos, CA, USA},
month = {may}
}
```

